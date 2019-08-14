from aienvs.Environment import Env
from aiagents.AgentComponent import AgentComponent


class AtomicAgent(AgentComponent):
    """
    An atomic agent -- class to inherit from, not instantiate
    """

    def __init__(self, agentId:str, environment:Env, parameters:dict=None):
        self._agentId = agentId
        self._environment = environment
        self._parameters = parameters

    def getAgentId(self):
        return self._agentId
    
    def getEnvironment(self):
        return self._environment

    def getParameters(self):
        return self._parameters
