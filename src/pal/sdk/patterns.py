"""ReAct and Reflexion as native agent patterns.

Both are reasoning *patterns* (papers), not libraries with a single call to intercept, so Pal ships them as
ready-made native agents built only on SDK calls. Subclass or configure them to make your own agent.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pal.sdk.agent import AgentResult, NativeAgent

_FINAL = re.compile(r"Final Answer:\s*(.*)", re.S | re.I)
_ACTION = re.compile(r"^\s*Action:\s*([A-Za-z_][\w.-]*)\s*(.*)$", re.M)
_CUT = re.compile(r"\n\s*Observation:.*", re.S)

REACT_FORMAT = (
    "Solve the task step by step. In each reply write 'Thought: <short reasoning>' and then either "
    "'Action: <tool_name> {\"arg\": value}' (JSON arguments) or 'Final Answer: <answer>'. After an Action you "
    "will receive 'Observation: <result>'. Tools: {tools}.")


@dataclass
class ReActStep:
    thought: str
    tool: str
    args: dict
    observation: str


class ReActAgent(NativeAgent):
    """Reason, act, observe (Yao et al., 2023): the model alternates thoughts with tool calls until it answers."""

    max_steps = 6
    max_tokens = 512
    tools: list[str] | None = None        # tool names to offer; None = every tool the SDK has

    def run(self, task: str, attachments: list[dict] | None = None, session: str = "agent") -> AgentResult:
        names = self.tools if self.tools is not None else self.sdk.list_tools()
        role = self.spec.role + "\n" if self.spec else ""
        transcript: list[dict] = [{"role": "system", "content": role + REACT_FORMAT.replace("{tools}", ", ".join(names) or "none")},
                                  {"role": "user", "content": task}]
        steps: list[ReActStep] = []
        for _ in range(self.max_steps):
            text = _CUT.sub("", self.sdk.chat(transcript, max_tokens=self.max_tokens) or "")   # ignore invented observations
            final = _FINAL.search(text)
            if final:
                return AgentResult(final.group(1).strip(), steps=steps)
            act = _ACTION.search(text)
            if act is None:                       # no action and no marker: treat the reply as the answer
                return AgentResult(text.strip(), steps=steps)
            tool, rest = act.group(1), act.group(2).strip()
            try:
                args = json.loads(rest) if rest else {}
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                observation = str(self.sdk.call_tool(tool, **args))
            except Exception as e:  # noqa: BLE001  (a bad call becomes an observation the model can react to)
                args, observation = {}, f"error: {e}"
            thought = text.split("Action:")[0].replace("Thought:", "").strip()
            steps.append(ReActStep(thought, tool, args, observation))
            transcript += [{"role": "assistant", "content": text}, {"role": "user", "content": f"Observation: {observation}"}]
        return AgentResult("", ok=False, steps=steps, error=f"no final answer after {self.max_steps} steps")


@dataclass
class Trial:
    answer: str
    ok: bool
    feedback: str
    reflection: str = ""


Evaluator = Callable[[str, str], "bool | tuple[bool, str]"]


class ReflexionAgent(NativeAgent):
    """Try, evaluate, reflect, retry (Shinn et al., 2023).

    The actor (any native agent; plain model call by default) answers. An evaluator (your function, or the model
    judging itself) says pass or fail. On failure the model writes a short lesson, which is kept in memory and
    shown to the next attempt.
    """

    max_trials = 3
    max_tokens = 400

    def __init__(self, sdk: Any = None, kernel: Any = None, actor: NativeAgent | None = None,
                 evaluator: Evaluator | None = None):
        super().__init__(sdk, kernel)
        self.actor = actor
        self.evaluator = evaluator

    def _act(self, prompt: str) -> str:
        if self.actor is not None:
            return self.actor.run(prompt).answer
        return self.sdk.chat([{"role": "user", "content": prompt}], max_tokens=self.max_tokens)

    def _evaluate(self, task: str, answer: str) -> tuple[bool, str]:
        if self.evaluator is not None:
            r = self.evaluator(task, answer)
            return (r, "") if isinstance(r, bool) else (bool(r[0]), str(r[1]))
        verdict = self.sdk.chat([
            {"role": "system", "content": "Judge whether the answer correctly and completely solves the task. "
                                          "Reply 'PASS' or 'FAIL: <reason>'."},
            {"role": "user", "content": f"Task: {task}\nAnswer: {answer}"}], max_tokens=80) or ""
        return verdict.strip().upper().startswith("PASS"), verdict.strip()

    def run(self, task: str, attachments: list[dict] | None = None, session: str = "agent") -> AgentResult:
        lessons: list[str] = []
        trials: list[Trial] = []
        for _ in range(self.max_trials):
            prompt = task if not lessons else task + "\n\nLessons from earlier attempts:\n- " + "\n- ".join(lessons)
            answer = self._act(prompt)
            ok, feedback = self._evaluate(task, answer)
            trial = Trial(answer, ok, feedback)
            trials.append(trial)
            if ok:
                return AgentResult(answer, steps=trials)
            trial.reflection = (self.sdk.chat([
                {"role": "system", "content": "In one or two sentences say what went wrong and what to do differently."},
                {"role": "user", "content": f"Task: {task}\nAnswer: {answer}\nFeedback: {feedback}"}],
                max_tokens=120) or "").strip()
            lessons.append(trial.reflection)
            self.sdk.remember(trial.reflection, tags=["reflexion"])
        return AgentResult(trials[-1].answer, ok=False, steps=trials, error=f"not solved in {self.max_trials} trials")
