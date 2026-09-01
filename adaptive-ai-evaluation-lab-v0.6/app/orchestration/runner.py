from app.agents.runtime import AgentRuntime
from app.domain.contracts import ExperimentState


class ExperimentRunner:
    """Execute a declared sequence of agents against one shared state."""

    def __init__(self, runtime: AgentRuntime, agents: list) -> None:
        self.runtime = runtime
        self.agents = agents

    async def execute_step(self, database, state: ExperimentState) -> ExperimentState:
        for agent in self.agents:
            state = await self.runtime.execute(database, state, agent)
        return state
