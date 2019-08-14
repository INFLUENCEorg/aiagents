from ..LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from aiagents.AgentFactory import createAgent, classForName, classForNameTyped


class testComplexAgentComponent(LoggedTestCase):

    def test_smoke_no_agents(self):
        env = Mock()
        params = {}
        with self.assertRaises(Exception) as context:
            createAgent('aiagents.multi.ComplexAgentComponent.ComplexAgentComponent', 'root', env, params)
        self.assertEquals("key 'agents' containing dict is required in parameters {}" , str(context.exception))

    def test_smoke_agents_no_dict(self):
        env = Mock()
        params = {'agents':3}
        with self.assertRaises(Exception) as context:
            createAgent('aiagents.multi.ComplexAgentComponent.ComplexAgentComponent', 'root', env, params)
        self.assertEquals("key 'agents' containing dict is required in parameters {'agents': 3}", str(context.exception))

    def test_smoke_bad_parameters(self):
        env = Mock()
        params = {'agents':{'a':1}}
        with self.assertRaises(Exception) as context:
            createAgent('aiagents.multi.ComplexAgentComponent.ComplexAgentComponent', 'root', env, params)
        self.assertEquals("agent a must have dict value: {'agents': {'a': 1}}" , str(context.exception))
