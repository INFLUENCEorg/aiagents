from ..LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from aiagents.AgentFactory import createAgent, classForName, classForNameTyped
from gym import spaces

compoundclass = 'aiagents.multi.BasicComplexAgent.BasicComplexAgent'
randomclass = 'aiagents.single.RandomAgent.RandomAgent'

env = Mock()
env.action_space = spaces.Dict({'robot1':spaces.Discrete(4)})


class testComplexAgentComponent(LoggedTestCase):

    def test_smoke_no_agents(self):
        params = {}
        with self.assertRaises(Exception) as context:
          createAgent(env, params)
        self.assertEquals("Parameters must have key 'class' but got {}" , str(context.exception))

    def test_smoke_agents_no_dict(self):
        params = {'class':randomclass, 'parameters':{}, 'subAgentList':3}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("the subAgentList parameter must contain a list, but got {'class': 'aiagents.single.RandomAgent.RandomAgent', 'parameters': {}, 'subAgentList': 3}", str(context.exception))

    def test_smoke_bad_parameters(self):
        params = {'class':randomclass, 'parameters':{}, 'subAgentList':{'a':1}}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("the subAgentList parameter must contain a list, but got {'class': 'aiagents.single.RandomAgent.RandomAgent', 'parameters': {}, 'subAgentList': {'a': 1}}" , str(context.exception))

    def test_smoke_empty_settings(self):
        params = {'class':randomclass, 'parameters':{}, 'subAgentList':{'a':{}}}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("the subAgentList parameter must contain a list, but got {'class': 'aiagents.single.RandomAgent.RandomAgent', 'parameters': {}, 'subAgentList': {'a': {}}}" , str(context.exception))

    def test_smoke_incomplete_settings(self):
        params = {'class':'some.class'}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("Parameters must have key 'parameters' containing a dict but got {'class': 'some.class'}" , str(context.exception))

    def test_smoke_bad_parameters_settings(self):
        params = {'class':'some.class', 'parameters':1}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("Parameters must have key 'parameters' containing a dict but got {'class': 'some.class', 'parameters': 1}" , str(context.exception))

    def test_smoke_no_some_class(self):
        params = {'class':'some.class', 'id':'robot1', 'parameters':{}}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("Can't load some.class from {'class': 'some.class', 'id': 'robot1', 'parameters': {}}" , str(context.exception))

    def test_smoke(self): 
        params = {'class':randomclass, 'id':'robot1', 'parameters':{}}
        createAgent(env, params)
        
    def test_check_subparty_rootclass_nochildren(self): 
        params = {'class':randomclass, 'parameters':{}, 'subAgentList':[{'class':randomclass, 'id':'robot1', 'parameters':{}}]}
        with self.assertRaises(Exception) as context:
            createAgent(env, params)
        self.assertEquals("aiagents.single.RandomAgent.RandomAgent failed on __init__:" , str(context.exception))

    def test_check_subparty(self): 
        params = {'class':compoundclass, 'parameters':{}, 'subAgentList':[{'class':randomclass, 'id':'robot1', 'parameters':{}}]}
        agt = createAgent(env, params)
        subs = agt._agentSubcomponents
        self.assertEquals(1, len(subs))
        self.assertEquals('RandomAgent', type(subs[0]).__name__)
