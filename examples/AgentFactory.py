from aiagents.AgentFactory import createAgent
from mock import Mock
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.utils import getParameters
import sys
import os



def main():
    """
    Demo how to construct agents
    """
    configName = "configs/agentconfig.yaml"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, configName)

    parameters=getParameters(filename)
 
    env = FactoryFloor(parameters['environment'])
    obs = env.reset()

    complexAgent = createAgent(env, parameters['agents'])
    print(complexAgent)
    print(complexAgent._agentSubcomponents)
    print(complexAgent._agentSubcomponents[2]._agentSubcomponents)
	
if __name__ == "__main__":
        main()
	

