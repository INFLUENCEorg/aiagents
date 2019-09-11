import sys
import os
import unittest
import logging
from test.LoggedTestCase import LoggedTestCase
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aiagents.single.PPO.PPOAgent import PPOAgent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.mcts.MctsAgent import MctsAgent
from aiagents.AgentComponent import AgentComponent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
import random
import yaml
from numpy import array
from numpy import ndarray
from aienvs.Environment import Env
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.utils import getParameters
from unittest.mock import Mock
from aienvs.loggers.JsonLogger import JsonLogger
import io
from aienvs.loggers.PickleLogger import PickleLogger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class testFactoryFloor(LoggedTestCase):
    """
    This is an integration test that also tests aiagents.
    """

    def test_mcts_agent(self):
        """
        Really a smoketest/demo how to run an agent
        """
        logging.info("Starting test_mcts_agent")
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../configs/factory_floor_simple.yaml")
        parameters = getParameters(filename)
        env = FactoryFloor(parameters)
        obs = env.reset()

        mctsAgents = []

        randomagent = 'aiagents.single.RandomAgent.RandomAgent'
        for robotId in env.action_space.spaces.keys():
            params = {'treeAgent':{'class': randomagent, 'id':robotId, 'parameters':{} },
                      'rolloutAgent':{'class': randomagent, 'id':robotId, 'parameters':{} }} 
            mctsAgents.append(MctsAgent(agentId=robotId, environment=env, parameters=params))

        complexAgent = BasicComplexAgent(mctsAgents)

        episode = Episode(complexAgent, env, obs, render=True)
        episode.run()

    def test_random_agent(self):
        logging.info("Starting test random agent")
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../configs/factory_floor_simple.yaml")
        parameters = getParameters(filename)

        env = FactoryFloor(parameters)
        randomAgents = []
        for robotId in env.action_space.spaces.keys():
            randomAgents.append(RandomAgent(robotId, env))

        complexAgent = BasicComplexAgent(randomAgents)
        steps = 0

        Experiment(complexAgent, env, 1000, None, True).run()

    def est_PPO_agent(self):
        logging.info("Starting test PPO agent")
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../configs/factory_floor_simple.yaml")
        parameters = getParameters(filename)

        env = FactoryFloor(parameters)
        PPOAgents = []
        for robotId in env.action_space.spaces.keys():
            PPOAgents.append(PPOAgent(parameters, env.observation_space, env.action_space.spaces.get(robotId), robotId))

        complexAgent = BasicComplexAgent(PPOAgents)
        steps = 0

        while steps < parameters["max_steps"]:
            steps += Episode(complexAgent, env, env.action_space.sample()).run()
        

if __name__ == '__main__':
    unittest.main()
