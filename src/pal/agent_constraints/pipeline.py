from pal.agent_constraints.components.constraints import load_agent_constraints


class AgentConstraintsPipeline:
    def with_constraints(self, system_prompt: str) -> str:
        """Merges the permanent output-behavior constraints into a given
        system prompt. Idempotent-ish: if the constraints text is
        already present (e.g. called twice on the same prompt), it
        isn't duplicated."""
        constraints = load_agent_constraints()
        if not constraints:
            return system_prompt
        if constraints in system_prompt:
            return system_prompt
        return f"{system_prompt}\n\n{constraints}"


# Module-level convenience, mirroring the old with_agent_constraints()
# free function so callers that just need the one operation don't have
# to instantiate the pipeline.
def with_agent_constraints(system_prompt: str) -> str:
    return AgentConstraintsPipeline().with_constraints(system_prompt)
