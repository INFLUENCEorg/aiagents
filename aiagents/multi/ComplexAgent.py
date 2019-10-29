from aiagents.AgentComponent import AgentComponent
from aienvs.Environment import Env


class ComplexAgent(AgentComponent):
    """
    A abstract class to inherit from, a complex agent component characterizes
    itself by being a collection of agent subcomponents.
    This is abstract because step is not implemented at this point.
    """

    def __init__(self, agentComponentList:list, environment:Env=None, parameters=None):
        """
        @param agentComponentList list of AgentComponent's children of this 
        @param environment the environment that this component and children work in
        @param parameters the additional setup parameters, used for
        this only (not for the children). 
        """
        self._agentSubcomponents = agentComponentList
        self._environment = environment

