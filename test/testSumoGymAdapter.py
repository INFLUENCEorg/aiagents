import sys
import os
import unittest
import aienvs
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.PPO.PPOAgent import PPOAgent
import logging
import yaml
from test.LoggedTestCase import LoggedTestCase
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
from aienvs.runners.Experiment import Experiment

logger = logging.getLogger()
logger.setLevel(50)


class testSumoGymAdapter(LoggedTestCase):
    """
    This is an integration test that tests both aienvs and aiagents.
    """

    def test_random_agent(self):
        logging.info("Starting test_random_agent")
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "configs/new_traffic_loop_ppo.yaml")

        with open(filename, 'r') as stream:
            try:
                parameters = yaml.safe_load(stream)['parameters']
            except yaml.YAMLError as exc:
                logging.error(exc)

        env = SumoGymAdapter(parameters)
        env.reset()

        randomAgents = []
        for intersectionId in env.action_space.spaces.keys():
            randomAgents.append(RandomAgent(intersectionId, env.action_space, env.observation_space))

        complexAgent = BasicComplexAgent(randomAgents, env.action_space, env.observation_space)
        experiment = Experiment(complexAgent, env, parameters['max_steps'])
        experiment.run()

    def test_PPO_agent(self):
        logging.info("Starting test_PPO_agent")
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "configs/new_traffic_loop_ppo.yaml")

        with open(filename, 'r') as stream:
            try:
                parameters = yaml.safe_load(stream)['parameters']
            except yaml.YAMLError as exc:
                logging.error(exc)

        env = SumoGymAdapter(parameters)
        env.reset()

        PPOAgents = []
        for intersectionId in env.action_space.spaces.keys():
            PPOAgents.append(PPOAgent(intersectionId, env.action_space, env.observation_space, parameters))

        complexAgent = BasicComplexAgent(PPOAgents, env.action_space, env.observation_space)
        experiment = Experiment(complexAgent, env, parameters['max_steps'])
        experiment.run()

        
if __name__ == '__main__':
    unittest.main()
    
