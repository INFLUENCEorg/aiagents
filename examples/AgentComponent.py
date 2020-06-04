from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aiagents.single.RandomAgent import RandomAgent
from gym.spaces import Discrete, Dict
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():

    N_agents = 10
    i = 0
    action_space = Dict({ n: Discrete(3) for n in range(N_agents) })
    obs_space = None  # not used anyway in this demo
    simpleComponentList = []

    while(i < N_agents):
        simpleComponentList.append(RandomAgent(i, action_space, obs_space))
        i += 1

    myComplexComponent = BasicComplexAgent(simpleComponentList, action_space, obs_space)

    N_steps = 3
    i = 0
    state = 0

    while i < N_steps:
        myComplexComponent.step(state)
        i += 1
    return 0

  
if __name__ == "__main__":
  main()

