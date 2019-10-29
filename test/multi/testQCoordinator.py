

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym import spaces
from gym.spaces import  Dict
from aiagents.QAgentComponent import QAgentComponent
from aiagents.multi.QCoordinator import QCoordinator


class testQCoordinator(LoggedTestCase):

    def test_initSmoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgentComponent)
        QCoordinator([component1], env)

    def test_initNoEnv(self):
        component1 = Mock(spec=QAgentComponent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], None)
        self.assertEquals("'NoneType' object has no attribute 'action_space'" , str(context.exception))

    def test_initNoDictEnv(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Discrete(3)
        component1 = Mock(spec=QAgentComponent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], env)
        self.assertEquals("Environment must have a Dict actionspace but found Discrete(3)" , str(context.exception))

    def test_initEmptyDict(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({})

        component1 = Mock(spec=QAgentComponent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], env)
        self.assertEquals("There are no actions in the space" , str(context.exception))

    def test_step_smoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgentComponent)
        component1.getQ = Mock(return_value=3.14) 
        coordinator = QCoordinator([component1], env)
        coordinator.step()

    @staticmethod
    def maxAt24(args, action:Dict):
        if (action.get('a') == 2 and action.get('b') == 4):
            return 3.14
        return 1

    def test_step_find_max(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        
        component1 = Mock(spec=QAgentComponent)
        component1.getQ = Mock(side_effect=testQCoordinator.maxAt24)
        coordinator = QCoordinator([component1], env)
        bestAction = coordinator.step()
        self.assertEqual(2, bestAction['a'])
        self.assertEqual(4, bestAction['b'])
    
