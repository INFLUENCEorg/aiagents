

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym import spaces
from aiagents.QAgentComponent import QAgentComponent
from aiagents.multi.QCoordinator import QCoordinator


class testQCoordinator(LoggedTestCase):
    
    def testBagger(self):
        pass
  
    def test_initSmoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgentComponent)
        QCoordinator([component1], env)

