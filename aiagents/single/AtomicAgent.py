# from aienvs.Environment import Env
from aiagents.AgentComponent import AgentComponent
from gym.spaces import Dict


class AtomicAgent(AgentComponent):
    """
    An atomic agent -- class to inherit from, not instantiate
    @param switchId the id of the environment entity that this agent controls.
    @param actionspace the gym action_space, must be Dict as required by our Envs
    @param observationspace the gym observation_space. 
    @param parameters additional parameters for the agent. The meaning depends
    on the specific agent being instantiated.
    """

    def __init__(self, switchId:str, actionspace:Dict=None, observationspace=None, parameters:dict=None):
        self._agentId = switchId
        self._actionspace = actionspace
        self._observationspace = observationspace

    @property
    def agentId(self):
        return self._agentId

    def getActionSpace(self):
        return self._actionspace
    
    def getObservationSpace(self):
        return self._observationspace
    
