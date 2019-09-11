from .LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from aiagents.AgentFactory import createAgent, classForName, classForNameTyped, resolve
import datetime
# import os
from gym.spaces import Dict, Discrete


class testAgentFactory(LoggedTestCase):
    """
    This is an integration test that also tests aiagents.
    aiagents project must be installed 
    """
    
    def test_get_class(self):
        D = classForName("datetime.datetime")
        time = D.now()
        actualtime = datetime.datetime.now()
        self.assertEquals(str(time)[0:10], str(actualtime)[0:10])

    def test_get_class_typed(self):
        D = classForNameTyped("datetime.datetime", datetime.datetime)
        time = D.now()
        actualtime = datetime.datetime.now()
        self.assertEquals(str(time)[0:10], str(actualtime)[0:10])

    def test_get_class_typed_wrong(self):
        self.assertRaises(Exception, classForNameTyped, "datetime.datetime", Env)

    def test_smoke(self):
        actions = Mock(spec=Discrete)
        actions.n = 10
        env = Mock()
        actspace = Mock(spec=Dict)
        spacesmock = { 'entity1':actions}
        actspace.spaces = spacesmock
        env.action_space = actspace

        agent = createAgent(env, {'id': 'entity1',
                                  'class':'aiagents.single.RandomAgent.RandomAgent',
                                  'parameters':{}})

    def test_is_good_agent(self):
        actions = Mock(spec=Discrete)
        actions.n = 10
        env = Mock()
        actspace = Mock(spec=Dict)
        spacesmock = { 'entity1':actions}
        actspace.spaces = spacesmock
        env.action_space = actspace
        state = Mock()
        
        agent = createAgent(env, {'id': 'entity1',
                                  'class':'aiagents.single.RandomAgent.RandomAgent',
                                  'parameters':{}})
        action = agent.step(state)
        print('agent did action:' + str(action))
        self.assertTrue(['entity1'], action.keys())

    def test_resolve_nonexisting(self):
        self.assertRaises(Exception, resolve, 'UnknownAgent', 'aiagents')
    
    def test_resolve_RandomAgent(self):
        self.assertEquals('aiagents.single.RandomAgent.RandomAgent', resolve('RandomAgent', 'aiagents'))
        # self.assertEquals('aiagents.single.RandomAgent.RandomAgent', resolve('RandomAgent', os.environ['AIAGENTS_HOME'] + '/aiagents'))
         
