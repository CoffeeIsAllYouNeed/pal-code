from dataclasses import dataclass

from pal.orchestrator.components.agent import Agent, AgentConfig
from pal.orchestrator.components.cgroup_manager import CgroupManager


@dataclass
class ScheduledAgent:
    config: AgentConfig
    priority: int = 0   # higher runs first when slots are contended
    weight: int = 100    # relative cpu.weight if backend supports it


class Scheduler:
    """
    Bounds how many agents run concurrently (max_concurrent), and
    decides which queued agents get a slot next based on priority.
    Doesn't touch cgroup internals directly -- that stays
    CgroupManager's job. This is purely admission control: who gets to
    start, and in what order.
    """

    def __init__(self, cgroup_manager: CgroupManager, max_concurrent: int = 3):
        self.cgroup_manager = cgroup_manager
        self.max_concurrent = max_concurrent
        self.running: dict[str, Agent] = {}
        self.queue: list[ScheduledAgent] = []

    def submit(self, config: AgentConfig, priority: int = 0, weight: int = 100) -> None:
        if config.name in self.running:
            raise ValueError(f"agent {config.name} already running")
        if any(s.config.name == config.name for s in self.queue):
            raise ValueError(f"agent {config.name} already queued")
        self.queue.append(ScheduledAgent(config=config, priority=priority, weight=weight))
        self.queue.sort(key=lambda s: s.priority, reverse=True)

    def tick(self) -> list[str]:
        """
        Admits as many queued agents as available slots allow, in
        priority order. Returns the names of agents newly started this
        tick. Call this after submit(), and again after any stop() to
        backfill freed slots.
        """
        started = []
        while self.queue and len(self.running) < self.max_concurrent:
            scheduled = self.queue.pop(0)
            agent = Agent(scheduled.config, self.cgroup_manager)
            agent.start()
            self.running[scheduled.config.name] = agent
            started.append(scheduled.config.name)
        return started

    def stop(self, agent_name: str) -> None:
        agent = self.running.pop(agent_name, None)
        if agent is None:
            raise ValueError(f"agent {agent_name} not running")
        agent.stop()

    def stop_all(self) -> None:
        for name in list(self.running.keys()):
            self.stop(name)

    def status(self) -> dict:
        return {
            "running": list(self.running.keys()),
            "queued": [s.config.name for s in self.queue],
            "capacity": self.max_concurrent,
            "slots_free": self.max_concurrent - len(self.running),
        }
