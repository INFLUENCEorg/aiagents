from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.QAgentComponent import QAgentComponent
import random

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
        for action in self._actionScope:
            q_dict.update({action: self.getQ(self._observation,action)})
        return {self._agentId: max(q_dict, key=q_dict.get)}

def main():

    N_agents=10
    i=0
    scope=[1,2,3]
    simpleComponentList=[]
    verbose=True

    while( i < N_agents ):
        simpleComponentList.append(RandomQAgent(i, scope, verbose))
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

