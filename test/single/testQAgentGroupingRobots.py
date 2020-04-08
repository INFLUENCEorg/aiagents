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
from test.LoggedTestCase import LoggedTestCase

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENTITY1 = "entity1"
ENTITY2 = "entity2"
ENTITY3 = "entity3"


class testQAgentGroupingRobot(LoggedTestCase):
    """
    Run the GroupingRobots environment with BasicComplexAgent and QCoordinator.
    Run 10 rounds only and print the maps and logged data at the end.
    Also demonstrates how BasicComplexAgent, PackedSpace, listener,
    Episode, QAgent and RandomAgent can work togeter in the GroupingRobots
    scene.
    """

    def test_Run(self):

        params = {'steps':10,
                    'robots':[ {'id': ENTITY1, 'pos':'random'},
                              {'id': ENTITY2, 'pos': 'random'},
                              {'id': ENTITY3, 'pos': 'random'}],  # initial robot positions
                    'seed':None,
                    'map':['....',
                           '....',
                           '....',
                           '....']
                    }

        basicEnv = GroupingRobots(params)
        logoutput = io.StringIO("episode output log")
        packedActionSpace = PackedSpace(basicEnv.action_space, {"e1":[ENTITY1], "e23": [ENTITY2, ENTITY3]})
        env = ModifiedGymEnv(basicEnv, packedActionSpace)

        agent1 = RandomAgent("e1", env.action_space, env.observation_space)
        agent2 = QAgent("e23", env, {'alpha':0.4, 'gamma':1, 'epsilon':0.1})
        complexAgent = BasicComplexAgent([agent1, agent2], basicEnv.action_space, basicEnv.observation_space)

        episode = Episode(complexAgent, env, None, True, 0)
        episode.addListener(JsonLogger(logoutput))
        episode.run()

        print("json output:", logoutput.getvalue())  # logs from all episodes within the experiment

