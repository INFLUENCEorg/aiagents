import os
import logging
from aiagents.single.PPO.PPOAgent import PPOAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
import random
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.utils import getParameters
from aienvs.loggers.JsonLogger import JsonLogger
import io
from aienvs.loggers.PickleLogger import PickleLogger
import copy
import sys
from aienvs.GroupingRobots.GroupingRobots import GroupingRobots
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.QAgent import QAgent
from aienvs.gym.PackedSpace import PackedSpace
from aienvs.gym.ModifiedGymEnv import ModifiedGymEnv

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
    
    learnEpisode()
    print("done")


def learnEpisode():
    basicEnv = GroupingRobots(params)
    packedActionSpace = PackedSpace(basicEnv.action_space, {"e1":[ENTITY1], "e23": [ENTITY2, ENTITY3]})
    env = ModifiedGymEnv(basicEnv, packedActionSpace)
    agent1 = RandomAgent("e1", env)
    agent2 = QAgent("e23", env, LEARN_PARAMS)
    complexAgent = BasicComplexAgent([agent1, agent2])
    episode = Episode(complexAgent, env, None, False, 0)
    episode.run()
    
    print("learned episode")

	
if __name__ == "__main__":
        main()
	
