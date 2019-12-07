import os
import logging
from aiagents.single.WarehouseNaiveAgent import WarehouseNaiveAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
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
    """
    Demo how to run an agent
    """
    # if( len(sys.argv) > 1 ):
    #     configName = str(sys.argv[1])
    #     filename = configName
    # else:
    #     print("Default config ")
    #     configName = "./configs/new_traffic_loop_ppo.yaml"
    #     dirname = os.path.dirname(__file__)
    #     filename = os.path.join(dirname, configName)

    # print( "Config name " + configName )
    logging.info("Starting example random agent")
    logoutput = io.StringIO("episode output log")
    # parameters = getParameters(filename)

    env = Warehouse()
    # here we initialize all agents (in that case one)
    randomAgents = []
    env.reset()
    for agent_id in env.action_space.spaces.keys():
        randomAgents.append(WarehouseNaiveAgent(robot_id=agent_id, env=env))
    complexAgent = BasicComplexAgent(randomAgents)
    experiment = Experiment(complexAgent, env, maxSteps=100, render=True, renderDelay=0.5)
    experiment.addListener(JsonLogger(logoutput))
    experiment.run()

    #print("json output:", logoutput.getvalue()) # logs from all episodes within the experiment

if __name__ == "__main__":
        main()
