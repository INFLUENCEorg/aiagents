from aiagents.AgentComponent import AgentComponent
from abc import abstractmethod
from math import inf


class QAgentComponent(AgentComponent):
    """
    Represents a Q-agent component
    In addition to the methods of a regular agent component 
    it has a method .getQ(state, action) and .getV(state)
    which returns the Q/V values predicted by the agent's internal model
    (V(state)=max over actions Q(state,action) in principle but could be evaluated by e.g. a neural network)
    """

    @abstractmethod
    def getQ(self, state, action) -> float:
        """
        @param state the current state / observation of the environment 
        @param action a dict describing an action, eg {"entityId2": a2, "entityId3": a3 }
        This dict may also contain actions for entities that are irrelevant
        for this component and have to be ignored then by the component. 
        @return Q value float of a <state,action> pair. This must not return
        +-inf.
        """
        pass

    @abstractmethod
    def getV(self, state):
        "V value of a state"
        pass

