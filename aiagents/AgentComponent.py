from abc import ABC, abstractmethod  # abstract base class
from gym import spaces
from aienvs.Environment import Env


class AgentComponent(ABC):
    """
    Represents an agent component -- a basic unit of decision making
    By stepping, agents observe the state and select their actions
    Agent components don't have to get rewards (not every agent component is an RL algorithm),
    but if they do, this can be handled within observing the state
    """

    @abstractmethod
    def step(self, observation=None, reward:float=None, done:bool=None) -> spaces.Dict:
        """
        should return a dictionary of controller ids and respective actions
        """
        pass

    @abstractmethod
    def __init__(self, agentId:str, environment:Env, parameters:dict=None):
        '''
        @param agentId the ID of the "entity"/"object" in the Env that 
        is controlled by this agent. Is assumed to be unique, that
        the agent controls only 1 entity. FIXME this assumption is not holding
        for complex/non-atomic agents.
        @param environment the Env that this agent works in.
        @param parameters the initialization dictionary parameters. For
        complex agents, these can also contain parameters for sub-agents
        '''
        pass
