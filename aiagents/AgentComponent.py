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

