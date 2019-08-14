from ..LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from aiagents.AgentFactory import createAgent, classForName, classForNameTyped

compoundclass = 'aiagents.multi.ComplexAgentComponent.ComplexAgentComponent'


class testComplexAgentComponent(LoggedTestCase):

    def test_smoke_no_agents(self):
        env = Mock()
        params = {}
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("key 'agents' containing dict is required in parameters {}" , str(context.exception))

    def test_smoke_agents_no_dict(self):
        env = Mock()
        params = {'agents':3}
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("key 'agents' containing dict is required in parameters {'agents': 3}", str(context.exception))

    def test_smoke_bad_parameters(self):
        env = Mock()
        params = {'agents':{'a':1}}
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("agent a must have dict value: {'agents': {'a': 1}}" , str(context.exception))

    def test_smoke_empty_settings(self):
        env = Mock()
        params = {'agents':{'a':{}}}
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("key 'classname' containing string is required in parameters {'agents': {'a': {}}}" , str(context.exception))

    def test_smoke_incomplete_settings(self):
        env = Mock()
        params = {'agents':{'a':{'classname':'some.class'}}}
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("key 'parameters' containing dict is required in parameters {'agents': {'a': {'classname': 'some.class'}}}" , str(context.exception))

    def test_smoke_bad_parameters_settings(self):
        env = Mock()
        params = {'agents':
                  {'a':{'classname':'some.class', 'parameters':1}}
                }
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("key 'parameters' containing dict is required in parameters {'agents': {'a': {'classname': 'some.class', 'parameters': 1}}}" , str(context.exception))

    def test_smoke_no_some_class(self):
        env = Mock()
        params = {'agents':
                  {'a':{'classname':'some.class', 'parameters':{}}}
                }
        with self.assertRaises(Exception) as context:
            createAgent(compoundclass, 'root', env, params)
        self.assertEquals("Failed to create sub-agent a using {'agents': {'a': {'classname': 'some.class', 'parameters': {}}}}" , str(context.exception))
 
