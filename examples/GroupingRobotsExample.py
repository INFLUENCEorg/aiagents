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


def main():
    """
    Run the GroupingRobots environment with BasicComplexAgent and QCoordinator.
    """
    configName = "./configs/new_traffic_loop_ppo.yaml"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, configName)
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
    
    basicEnv = GroupingRobots()
    logoutput = io.StringIO("episode output log")
    packedActionSpace = PackedSpace(basicEnv.action_space, {"e1":[ENTITY1], "e23": [ENTITY2, ENTITY3]})
    env = ModifiedGymEnv(basicEnv, packedActionSpace)

    agent1 = RandomAgent("e1", env)
    agent2 = QAgent("e23", PackedSpace(env), {'alpha':0.4, 'gamma':1, 'm':-500000, 's':1})
    complexAgent = BasicComplexAgent([agent1, agent2])
    
    episode = Episode(complexAgent, env, None, True, 0)
    episode.addListener(JsonLogger(logoutput))
    episode.run()

    print("json output:", logoutput.getvalue())  # logs from all episodes within the experiment

	
if __name__ == "__main__":
        main()
	
