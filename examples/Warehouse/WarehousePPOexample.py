import os
import logging
from aiagents.single.WarehouseNaiveAgent import WarehouseNaiveAgent
from aiagents.single.PPO.PPOAgent import PPOAgent
from aiagents.single.PPO.vectorizedEnvironment import VectorizedEnvironment
from aiagents.multi.MultiAgent import MultiAgent
import random
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.Warehouse import Warehouse
from aienvs.utils import getParameters
from aienvs.loggers.JsonLogger import JsonLogger
import io
from aienvs.loggers.PickleLogger import PickleLogger
import copy
import sys
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    if( len(sys.argv) > 1 ):
        configName = str(sys.argv[1])
        filename = configName
    else:
        print("Default config ")
        configName = "../configs/PPO.yaml"
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, configName)

    logging.info("Starting example warehouse DQN agent")
    logoutput = io.StringIO("episode output log")
    parameters = getParameters(filename)

    env = Warehouse()
    # here we initialize all agents (in that case one)
    agents = []
    env.reset()
    for agent_id, action_space in env.action_space.spaces.items():
        agents.append(WarehouseNaiveAgent(robot_id=agent_id, actionspace=action_space, 
                                          observationspace=env.observation_space))
    breakpoint()
    agents[20] = PPOAgent(agentId=20, actionspace=env.action_space[20],
                          observationspace=env.observation_space,
                          parameters=parameters['all'])
    multi_agent = MultiAgent(agents)
    experiment = Experiment(multi_agent, env, maxSteps=1e6, render=True,
                            seedlist=[random.randrange(1, 1e3, 1) for i in range(int(1e5))],
                            renderDelay=0.5)
    experiment.addListener(JsonLogger(logoutput))
    experiment.run()

    #print("json output:", logoutput.getvalue()) # logs from all episodes within the experiment

if __name__ == "__main__":
    main()
