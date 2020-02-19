from aiagents.AgentComponent import AgentComponent
from aienvs.Environment import Env
from gym.spaces import Dict
from typing import List


class ComplexAgent(AgentComponent):
    """
    A abstract class to inherit from, a complex agent component characterizes
    itself by being a collection of agent subcomponents.
    This is abstract because step is not implemented at this point.
    """

    def __init__(self, agentComponentList:List[AgentComponent], actionspace:Dict, observationspace, parameters=None):
        """
        @param agentComponentList list of AgentComponent's children of this 
        @param actionspace the gym action_space, must be Dict as required by our Envs.
        Usually, this is an unpacked space, allowing the complex agent to 
        generate un-packed actions and then pack them properly for each AgentComponent
        @param observationspace the gym observation_space. 
        @param parameters the additional setup parameters, used for
        this only (not for the children). 
        """
        self._agentSubcomponents = agentComponentList
        self._actionspace = actionspace
        self._observationspace = observationspace

