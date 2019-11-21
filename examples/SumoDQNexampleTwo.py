import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
from aiagents.single.DQN.DQNAgent import DQNAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
import random
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.utils import getParameters
from aienvs.loggers.JsonLogger import JsonLogger
import io
from aienvs.loggers.PickleLogger import PickleLogger
from aienvs.gym.PackedSpace import PackedSpace
from aienvs.gym.ModifiedGymEnv import ModifiedGymEnv
from aienvs.gym.DecoratedSpace import DecoratedSpace
import copy
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def main():
    """
    Demo how to run an agent
    """
    if( len(sys.argv) > 1 ):
        configName = str(sys.argv[1])
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, configName)
    else:
        print("Default config ")
        configName = "./configs/two_grid_H_DQN_Debugging.yaml"
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, configName)

    print( "Config name " + configName )
    logging.info("Starting example DQN agent")

    parameters = getParameters(filename)

    env = SumoGymAdapter(parameters['all'])

    # create a joint action space for the agent that controls two-intersection
    packed_action_space = PackedSpace(env.action_space, {"joint":["0", "1"]})
    env = ModifiedGymEnv(env, packed_action_space)

    # here we initialize all agents (in that case one)
    Agents = []
    env.reset()
    for intersectionId in env.action_space.spaces.keys():
        Agents.append(DQNAgent(agentId=intersectionId, environment=env, parameters=parameters['all']))
    complexAgent = BasicComplexAgent(Agents)

    experiment_name = parameters["all"]["name"] 
    log_stream = open("results/{}.json".format(experiment_name), "w", encoding="utf-8")

    experiment = Experiment(complexAgent, env, parameters['all']['max_steps'], parameters['all']['seedlist'], render=False)
    experiment.addListener(JsonLogger(log_stream))
    experiment.run()

    log_stream.close()

    # print("json output:", logoutput.getvalue()) # logs from all episodes within the experiment

if __name__ == "__main__":
        main()
