import os
import io
import copy
import sys
import logging
import random

from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.utils import getParameters
from aienvs.loggers.JsonLogger import JsonLogger
from aienvs.loggers.PickleLogger import PickleLogger
from aienvs.GroupingRobots.GroupingRobots import GroupingRobots
from aienvs.gym.PackedSpace import PackedSpace
from aienvs.gym.ModifiedGymEnv import ModifiedGymEnv

from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.QAgent import QAgent
from aiagents.multi.QCoordinator import QCoordinator
from aiagents.single.PPO.PPOAgent import PPOAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENTITY1 = "entity1"
ENTITY2 = "entity2"
ENTITY3 = "entity3"

params = {'steps':1000,
            'robots':[ {'id': ENTITY1, 'pos':'random'},
                      {'id': ENTITY2, 'pos': 'random'},
                      {'id': ENTITY3, 'pos': 'random'}],  # initial robot positions
            'seed':None,
            'map':['....',
                   '....',
                   '....',
                   '....']
            }
LEARN_PARAMS = {'alpha':0.4, 'gamma':1, 'm':-500000, 's':1}


def main():
    """
    This demo contains of 4 episode runs:
    1. run new QAgent learning combined actions of robot1,2
    2. run new QAgent learning combined actions of robot1,3
    3. run new QAgent learning combined actions of robot2,3
    4. run QCoordinator using the 3 QAgents 
    
    In step 1..3 we also have a RandomAgent doing random actions
    for the remaining robot.
    """
    
    qagent1 = learnEpisode(ENTITY1)
    qagent2 = learnEpisode(ENTITY2)
    qagent3 = learnEpisode(ENTITY3)
    print("done learning, running the QCoordinator")

    runQCoordinator(qagent1, qagent2, qagent3)
    print("finished the example")
    

def learnEpisode(randomentity:str) -> QAgent:
    '''
    @param randomentity the entity controlled by the randomagent.
    Should be one of {ENTITY1,ENTITY2, ENTITY3}.
    The remaining entities will be controlled by the QAgent.
    @return the trained QAgent.
    '''
    qagententities = [ENTITY1, ENTITY2, ENTITY3]
    qagententities.remove(randomentity)
    
    basicEnv = GroupingRobots(params)
    packedActionSpace = PackedSpace(basicEnv.action_space, \
                {"random":[randomentity], "qlearn": qagententities})
    env = ModifiedGymEnv(basicEnv, packedActionSpace)
    agent1 = RandomAgent("random", env.action_space, env.observation_space)
    agent2 = QAgent("qlearn", env.action_space, env.observation_space, LEARN_PARAMS)
    complexAgent = BasicComplexAgent([agent1, agent2], env.action_space, env.observation_space)
    episode = Episode(complexAgent, env, None, False, 0)
    episode.run()
    
    print("learned episode")
    return agent2


def runQCoordinator(qagent1, qagent2, qagent3):
    '''
    Create an episode, now with plain environment.
    The agent is now the QCoordinator consisting of QAgent1, QAgent2, QAgent3.
    The agents are trained on a different PackedSpace of our environment. 
    The QCoordinator will query the QAgents for Q values 
    and should choose the best perceived joint action based on them..
    '''
    env = GroupingRobots(params)
    qcoord = QCoordinator([qagent1, qagent2, qagent3], env.action_space, env.observation_space)
    episode = Episode(qcoord, env, None, False, 0)
    episode.run()

	
if __name__ == "__main__":
        main()
	
