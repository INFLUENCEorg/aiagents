from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.QAgentComponent import QAgentComponent
import random
from gym.spaces import Discrete

class RandomQAgent(RandomAgent, QAgentComponent):
    """
    A dummy Q agent component
    where the look ahead horizon is 1, so only the immediate return is used
    """
    def getQ(self,state,action):
        """
        1 if state matches action 0 otherwise
        """
        return state==action

    def getV(self, state):
        """
        max over Q(state,action)
        """
        q_list=[]
        for action in self._actionScope:
            q_list.append(self.getQ(state,action))
        return max(q_list)

    def select_actions(self):
        """
        We select action that maximizes Q
        """
        q_dict=dict()
        actionId = 0
    
        while self._actionSpace.contains(actionId):
            q_dict.update({actionId: self.getQ(self._observation,actionId)})
            actionId+=1
            print(actionId)
        return {self._agentId: max(q_dict, key=q_dict.get)}

def main():

    N_agents=10
    i=0
    action_space=Discrete(3)
    simpleComponentList=[]
    verbose=True

    while( i < N_agents ):
        simpleComponentList.append(RandomQAgent(i, action_space))
        i+=1

    myComplexComponent=ComplexAgentComponent(simpleComponentList, verbose)

    N_steps=6
    i=0
    state=1

    while i<N_steps:
        state = (state+1) % 4
        print("State = " + str(state))
        myComplexComponent.observe(state)
        myComplexComponent.select_actions()
        i+=1
    return 0
  
if __name__== "__main__":
  main()

