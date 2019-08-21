from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.QAgentComponent import QAgentComponent
import random
from gym.spaces import Discrete
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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

    def step(self, observation, reward, done):
        """
        We select action that maximizes Q
        """
        q_dict=dict()
        actionId = 0
    
        while self._actionSpace.contains(actionId):
            q_dict.update({actionId: self.getQ(observation,actionId)})
            actionId+=1
            logging.debug("ACTION: " + str(actionId))
        return {self._agentId: max(q_dict, key=q_dict.get)}

def main():

    N_agents=10
    i=0
    action_space=Discrete(3)
    simpleComponentList=[]

    while( i < N_agents ):
        simpleComponentList.append(RandomQAgent(i, action_space))
        i+=1

    myComplexComponent=BasicComplexAgent(simpleComponentList)

    N_steps=6
    i=0
    state=1

    while i<N_steps:
        state = (state+1) % 4
        print("State = " + str(state))
        myComplexComponent.step(state)
        i+=1
    return 0
  
if __name__== "__main__":
  main()

