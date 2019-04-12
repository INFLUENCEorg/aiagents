from abc import ABC, abstractmethod #abstract base class

class AgentComponent(ABC):
    """
    Represents an agent component -- a basic unit of decision making
    Agent components can observe the state
    Agent components can select actions
    Agent components keep track of steps of execution of the program
    Agent components don't have to get rewards (not every agent component is an RL algorithm),
    but if they do, this can be handled within observing the state
    """
    @abstractmethod
    def observe(self, state):
        """
        no return
        """
        pass

    @abstractmethod
    def select_actions(self):
        """
        should return a dictionary of agent ids and actions
        TODO: how to enforce this?
        """
        pass

