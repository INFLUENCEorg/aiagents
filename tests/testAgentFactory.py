from LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from aiagents.AgentFactory import createAgent, classForName
import datetime


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

    def test_smoke(self):
        env = Mock()
        agent = createAgent('aiagents.single.RandomAgent.RandomAgent', 'entity1', env, {})

    def test_is_good_agent(self):
        actions = Mock()
        actions.n = 10
        env = Mock()
        env.action_space.spaces.get.return_value = actions
        state = Mock()
        
        agent = createAgent('aiagents.single.RandomAgent.RandomAgent', 'entity1', env, {})
        action = agent.step(state)
        print('agent did action:' + str(action))
        self.assertTrue(['entity1'], action.keys())
        
