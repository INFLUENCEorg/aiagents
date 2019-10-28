from aiagents.multi.ComplexAgent import ComplexAgent
from .BasicComplexAgent import BasicComplexAgent
from ..QAgentComponent import QAgentComponent
from gym import spaces
from aienvs.gym.DecoratedSpace import DecoratedSpace
from aienvs.Environment import Env
from route.routecompare import INFINITY


class QCoordinator(BasicComplexAgent):
    """
    The QCoordinator should work with environments that have a dictionary action space of the form:
    A={
    "entityId1": Discrete(n1)
    "entitityId2": Discrete(n2)
    "entityId3": Discrete(n3)
    ... 
    }
    Discrete(n) is the gym Discrete action space on n actions.
    
    """

    def __init__(self, agentComponentList:list, environment:Env, parameters:dict=None):
        """
        @param AgentComponentList: a list of QAgentComponent.
        @param environment the openAI Gym Env
        @param parameters the optional init dictionary with parameters  
        """
        if not isinstance(environment.action_space, spaces.Dict):
            raise ValueError("Environment must have a Dict actionspace but found " + environment.action_space)
        for component in agentComponentList:
            if not isinstance(component, QAgentComponent):
                raise ValueError("All agent components for QCoordinator must be QAgentComponent but found " + component)
        super(QCoordinator, self).__init__(agentComponentList, parameters)
        self.actionspace = DecoratedSpace.create(environment.action_space)
        if self.actionspace.getSize() == 0:
            # error because we then can't find the best action
            raise ValueError("There are no actions in the space")
    
    def step(self, observation=None, reward:float=None, done:bool=None) -> spaces.Dict:
        """
        Let's say QCoordinator has 3 QAgentComponent's, 
        lets denote them subQ1, subQ2, subQ3.
        The QCoordinator needs to decide which action to choose from the set $A$,
        that is, evaluate all possible combinations of integers (a1,a2,a3) 
        which correspond to a full joint action
        {
        "entityId1": a1
        "entityId2": a2
        "entityId3": a3
        }
        and chose the one it thinks its best. Below I describe how this action 
        will be chosen.
        
        It is assumed that each subcomponent can evaluate a partial joint action 
        by its method getQ, e.g. subQ1 can evaluate
        {
        "entityId1": a1
        "entityId2": a2
        }
        
        subQ2 can evaluate
        {
        "entityId2": a2
        "entityId3": a3
        }
        
        subQ3 can evaluate
        {
        "entityId2": a1
        "entityId3": a3
        }
        
        (other combinations, amount of subcomponents, amount of subactions in 
        subcomponents etc. are possible.)
        
        The rule for QCoordinator step:
        
        Loop over all possible (a1,a2,a3)
            return action ("entity1": a1, "entity2": a2, "entity3": a3) such that
        
                subQ1.getQ(observation, {"entity1": a1, "entity2": a2}) +
                subQ2.getQ(observation, {"entity2": a2, "entity3": a3}) +
                subQ3.getQ(observation, {"entity1": a1, "entity3": a3})
        
            is MAXIMAL.
        """
        bestTotalQ = float('-inf')
        bestAction = None
        for n in range(0, self.actionspace.getSize()):
            action = self.actionspace.getById(n)
            totalQ = 0
            for agentComp in self._agentSubcomponents:
                totalQ = totalQ + agentComp.getQ(observation, action)
            if totalQ > bestTotalQ:
                bestAction = action
                bestTotalQ = totalQ
        # bestAction can be None still only if all actions have Q=-inf
        return bestAction
