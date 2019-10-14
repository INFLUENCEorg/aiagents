from aiagents.single.AtomicAgent import AtomicAgent

class ConstantAgent(AtomicAgent):
    def __init__(self, agentId, environment, parameters):
        self._action = parameters["action"]
        super().__init__(agentId, environment, parameters)

    def step(self, observation, reward, done):
        return {self._agentId: action}
