from aienvs.Environment import Env
from aiagents.AgentComponent import AgentComponent


class AtomicAgent(AgentComponent):
    """
    An atomic agent -- class to inherit from, not instantiate
    """

    def __init__(self, switchId:str, environment:Env, parameters:dict=None):
        self._agentId = switchId
        self._environment = environment

    @property
    def agentId(self):
        return self._agentId

