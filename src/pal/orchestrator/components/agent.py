import multiprocessing as mp
from dataclasses import dataclass

from pal.orchestrator.components.cgroup_manager import CgroupManager, ResourceLimits


def _agent_worker(backend: str, model: str, host: str, task_q: mp.Queue, result_q: mp.Queue):
    """
    Runs inside the child process. Imports pal lazily, here, not
    at module level -- the parent process doesn't need pal
    importable to manage cgroups/scheduling, only the child that
    actually runs a Pipeline does. Keeps the orchestrator's own
    dependency footprint minimal.
    """
    from pal.llm_wrapper.components.base import WrapperConfig
    from pal.pipeline import Pipeline

    pipeline = Pipeline(WrapperConfig(backend=backend, model=model, host=host))

    while True:
        task = task_q.get()
        if task is None:
            break
        system_prompt, user_message = task
        try:
            result = pipeline.run(system_prompt, user_message)
            result_q.put(("ok", result, pipeline.summary()))
        except Exception as e:
            result_q.put(("error", str(e), None))


@dataclass
class AgentConfig:
    name: str
    backend: str = "ollama"
    model: str = "llama3"
    host: str = "http://localhost:11434"
    limits: ResourceLimits = None


class Agent:
    """
    One agent = one real OS process running a Pal Pipeline, bound
    to its own cgroup slice. The cgroup enforces the resource envelope;
    the process boundary is what makes that enforcement meaningful
    (unlike an in-process object, which shares the parent's resource
    pool).
    """

    def __init__(self, config: AgentConfig, cgroup_manager: CgroupManager):
        self.config = config
        self.cgroup_manager = cgroup_manager
        self.process: mp.Process | None = None
        self.task_q: mp.Queue = mp.Queue()
        self.result_q: mp.Queue = mp.Queue()
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"agent {self.config.name} already started")

        limits = self.config.limits or ResourceLimits(memory_max="512M", pids_max="50")
        self.cgroup_manager.create_slice(self.config.name, limits)

        self.process = mp.Process(
            target=_agent_worker,
            args=(self.config.backend, self.config.model, self.config.host, self.task_q, self.result_q),
            daemon=True,
        )
        self.process.start()

        # Assign immediately after spawn so the process's entire
        # lifetime -- including model/library imports and any threads it
        # spins up -- is subject to the cgroup's limits, not just the
        # work it does after some later point.
        self.cgroup_manager.assign_pid(self.config.name, self.process.pid)
        self._started = True

    def run_turn(self, system_prompt: str, user_message: str, timeout: float = 60) -> dict:
        if not self._started:
            raise RuntimeError(f"agent {self.config.name} not started")

        self.task_q.put((system_prompt, user_message))
        status, payload, summary = self.result_q.get(timeout=timeout)

        if status == "error":
            raise RuntimeError(f"agent {self.config.name} failed: {payload}")

        return {"response": payload, "metrics": summary}

    def stats(self) -> dict:
        return self.cgroup_manager.stats(self.config.name)

    def stop(self) -> None:
        if not self._started:
            return

        self.task_q.put(None)
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)

        self.cgroup_manager.delete_slice(self.config.name)
        self._started = False
