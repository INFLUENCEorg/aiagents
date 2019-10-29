from aiagents.AgentComponent import AgentComponent
from aienvs.Environment import Env


class ComplexAgent(AgentComponent):
    """
    A abstract class to inherit from, a complex agent component characterizes
    itself by being a collection of agent subcomponents.
    This is abstract because step is not implemented at this point.
    """

    def __init__(self, agentComponentList, environment:Env=None, parameters=None):
        self._agentSubcomponents = agentComponentList
        self._environment = environment

