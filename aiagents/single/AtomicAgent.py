from aienvs.Environment import Env
from aiagents.AgentComponent import AgentComponent


class AtomicAgent(AgentComponent):
    """
    An atomic agent -- class to inherit from, not instantiate
    @param switchId the id of the environment entity that this agent controls.
    @param environment the gym environment
    @param parameters additional parameters for the agent. The meaning depends
    on the specific agent being instantiated.
    """

    def __init__(self, switchId:str, environment:Env, parameters:dict=None):
        self._agentId = switchId
        self._environment = environment

    @property
    def agentId(self):
        return self._agentId

    def getEnvironment(self):
        return self._environment
