from aiagents.AgentFactory import createAgent
from unittest.mock import Mock
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.utils import getParameters
import sys
import os


def main():
    """
    Demo how to construct complex agents
    """
    configName = "configs/factory_floor_complex.yaml"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, configName)

    parameters = getParameters(filename)
 
    env = FactoryFloor(parameters['environment'])
    obs = env.reset()

    complexAgent = createAgent(env.action_space, env.observation_space, parameters['agents'])
    print(complexAgent)
    sub = complexAgent._agentSubcomponents[0]
    subsub = sub._agentSubcomponents
    print(subsub)

	
if __name__ == "__main__":
        main()

