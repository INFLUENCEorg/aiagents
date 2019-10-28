from aiagents.AgentComponent import AgentComponent


class ComplexAgent(AgentComponent):
    """
    A class to inherit from, a complex agent component characterizes
    itself by being a collection of agent subcomponents
    """

    def __init__(self, agentComponentList, parameters=None):
        self._agentSubcomponents = agentComponentList

    def getSubAgents(self) -> list:  # <AgentComponent>
        return self._agentSubcomponents
