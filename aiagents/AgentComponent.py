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
        Let's the agentcomponent decide on its action.
        This is called after the env.step() was called to get the observation and 
        reward. Learners may have to store their last action and state as well,
        so that they can connect the last and current state with the action
        and the reward.
        
        @param observation the current observation/state. This can be any object
        as long as its hash code and equals are equal for equal states.
        @param reward the reward received from *last action*
        @param done this is true if the environment reached the final state.

        @return an action: a dictionary of controller ids and respective actions
        """
        pass

