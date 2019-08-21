from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aiagents.single.RandomAgent import RandomAgent
from gym.spaces import Discrete
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def main():

    N_agents=10
    i=0
    action_space=Discrete(3)
    simpleComponentList=[]

    while( i < N_agents ):
        simpleComponentList.append(RandomAgent(i, action_space))
        i+=1

    myComplexComponent=BasicComplexAgent(simpleComponentList)

    N_steps=3
    i=0
    state=0

    while i<N_steps:
        myComplexComponent.step(state)
        i+=1
    return 0
  
if __name__== "__main__":
  main()

