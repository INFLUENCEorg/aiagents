

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym import spaces
from gym.spaces import  Dict
from aiagents.single.QAgent import QAgent
from aiagents.multi.QCoordinator import QCoordinator
from aienvs.gym.ModifiedActionSpace import ModifiedActionSpace
from aienvs.gym.PackedSpace import PackedSpace


class testQCoordinator(LoggedTestCase):

    def test_initSmoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgent)
        QCoordinator([component1], env)

    def test_initNoEnv(self):
        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], None)
        self.assertEquals("'NoneType' object has no attribute 'action_space'" , str(context.exception))

    def test_initNoDictEnv(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Discrete(3)
        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], env)
        self.assertEquals("Environment must have a Dict actionspace but found Discrete(3)" , str(context.exception))

    def test_initEmptyDict(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({})

        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], env)
        self.assertEquals("There are no actions in the space" , str(context.exception))

    def test_step_smoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        env.action_space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgent)
        component1.agentId = 'a'
        component1.getQ = Mock(return_value=3.14)
        component1.getEnvironment = Mock(return_value=env) 
        coordinator = QCoordinator([component1], env)
        coordinator.step()
        
    @staticmethod
    def maxAt2(args, action:Dict):
        if action.get('a') == 2:
            return 3.14
        return 1

    def test_step_find_max(self):
        # we don't want to test spaces but we need to get DecoratedSpace
        # so it seems easier to make a real space anyway.
        space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        env = Mock(spec=Env)
        env.action_space = space
        
        component1 = Mock(spec=QAgent)
        component1.agentId = 'a'  # the controlled entity
        component1.getQ = Mock(side_effect=testQCoordinator.maxAt2)
        component1.getEnvironment = Mock(return_value=env)

        coordinator = QCoordinator([component1], env)
        bestAction = coordinator.step()
        self.assertEqual(2, bestAction['a'])
        # we don't know B because there is no agent prefering any b value
    
    @staticmethod
    def maxAt24(args, action:Dict):
        if action.get('ab') == 24:
            return 3.14
        return 1
