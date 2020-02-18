from aiagents.multi.ComplexAgent import ComplexAgent
from .BasicComplexAgent import BasicComplexAgent
from aiagents.QAgentComponent import QAgentComponent
from gym import spaces
from aienvs.gym.DecoratedSpace import DecoratedSpace
from aienvs.Environment import Env
from aienvs.gym.ModifiedActionSpace import ModifiedActionSpace


class QCoordinator(BasicComplexAgent):
    """
    A QCoordinator is an agent that can coordinate a set of 
    sub-agents in such a way that the sum of their actions is optimal.
    See #step for more details.
    
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
        @param AgentComponentList: a list of QAgentComponent. Size must be >=1.
        The agent environments should be equal to our environment,
        or to a Packed version of it. We can't check this because
        environments do not implement equals at this moment.
        @param environment the openAI Gym Env. Must have actionspace of type Dict.
        Must be a non-packed space, so that the actions can be packed
        properly for each QAgentComponent individually.
        @param parameters the optional init dictionary with parameters  
        """
        if not isinstance(environment.action_space, spaces.Dict):
            raise ValueError("Environment must have a Dict actionspace but found " + str(environment.action_space))

        if len(agentComponentList) == 0:
            raise ValueError("There must be at least 1 agent in the list")
        
        for agent in agentComponentList:
            if not isinstance(agent, QAgentComponent):
                raise ValueError("All agent components for QCoordinator must be QAgentComponent but found " + agent)

        super(QCoordinator, self).__init__(agentComponentList, environment, parameters)
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
            for agent in self._agentSubcomponents:
                env = agent.getEnvironment().action_space
                # HACK #7, pack action if agents are using packed space
                if isinstance(env, ModifiedActionSpace):
                    action1 = env.pack(action)
                else:
                    action1 = action
                totalQ = totalQ + agent.getQ(observation, action1)
            if totalQ > bestTotalQ:
                bestAction = action
                bestTotalQ = totalQ
        # bestAction can be None still only if all actions have Q=-inf
        return bestAction
