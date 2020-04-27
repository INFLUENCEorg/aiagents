

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
        # env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        action_space = Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})

        component1 = Mock(spec=QAgent)
        QCoordinator([component1], action_space, None, {})

    def test_initNoEnv(self):
        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], None)
        self.assertEquals("actionspace must be Dict but found None" , str(context.exception))

    def test_initNoDictEnv(self):
        # we don't want to test spaces but we need to get DecoratedSpace
        space = spaces.Discrete(3)
        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], space, None)
        self.assertEquals("actionspace must be Dict but found Discrete(3)" , str(context.exception))

    def test_initEmptyDict(self):
        # we don't want to test spaces but we need to get DecoratedSpace
        space = spaces.Dict({})

        component1 = Mock(spec=QAgent)
        with self.assertRaises(Exception) as context:
            QCoordinator([component1], space, None)
        self.assertEquals("There are no actions in the space" , str(context.exception))

    def test_step_smoke(self):
        env = Mock(spec=Env)
        # we don't want to test spaces but we need to get DecoratedSpace
        space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        env.action_space = space

        component1 = Mock(spec=QAgent)
        component1.agentId = 'a'
        component1.getQ = Mock(return_value=3.14)
        component1.getEnvironment = Mock(return_value=env) 
        coordinator = QCoordinator([component1], space, None)
        coordinator.step()
        
    @staticmethod
    def maxAtA2(args, action:Dict):
        if action.get('a') == 2:
            return 3.14
        return 1

    @staticmethod
    def maxAtB4(args, action:Dict):
        if action.get('b') == 4:
            return 2.2
        return 0.3
    
    def test_step_find_max_one_agent(self):
        '''
        Test where QCoordinator has only 1 sub-agent, which prefers a=2.
        QCoordinator thus should find an action with a=2 (b is irrelevant).
        '''
        # we don't want to test spaces but we need to get DecoratedSpace
        # so it seems easier to make a real space anyway.
        space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        env = Mock(spec=Env)
        env.action_space = space
        
        component1 = Mock(spec=QAgent)
        component1.agentId = 'a'  # the controlled entity
        component1.getQ = Mock(side_effect=testQCoordinator.maxAtA2)
        component1.getEnvironment = Mock(return_value=env)

        coordinator = QCoordinator([component1], space, None)
        bestAction = coordinator.step()
        self.assertEqual(2, bestAction['a'])
        # we don't know B because there is no agent prefering any b value

    def test_step_find_max_two_agents(self):
        '''
        Test with QCoordinator that has two sub-agents.
        One prefers a=2, the other prefers b=4.
        QCoordinator should find the action with a=2, b=4.
        '''
        # we don't want to test spaces but we need to get DecoratedSpace
        # so it seems easier to make a real space anyway.
        space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        env = Mock(spec=Env)
        env.action_space = space
        
        componentA = Mock(spec=QAgent)
        componentA.agentId = 'a'  # the controlled entity
        componentA.getQ = Mock(side_effect=testQCoordinator.maxAtA2)
        componentA.getEnvironment = Mock(return_value=env)

        componentB = Mock(spec=QAgent)
        componentB.agentId = 'b'  # the controlled entity
        componentB.getQ = Mock(side_effect=testQCoordinator.maxAtB4)
        componentB.getEnvironment = Mock(return_value=env)

        coordinator = QCoordinator([componentA, componentB], space, None)
        bestAction = coordinator.step()
        self.assertEqual(2, bestAction['a'])
        self.assertEqual(4, bestAction['b'])
    
    @staticmethod
    def maxAtAB16(args, action:Dict):
        if action.get('ab') == 16:
            return 2.2
        return 0.3
    
    def test_step_find_max_one_packedagent(self):
        '''
        Test where QCoordinator has 1 sub-agent that uses a packed space,
        which prefers the action ab=24. 
        '''
        # we don't want to test spaces but we need to get DecoratedSpace
        # so it seems easier to make a real space anyway.
        space = spaces.Dict({'a':spaces.Discrete(3), 'b':spaces.Discrete(7)})
        # env = Mock(spec=Env)
        # env.action_space = space
        
        # we also grab this moment to demo packedspace again.
        # for proper junit test this should be mocked.
        packedspace = PackedSpace(space, {'ab':['a', 'b']})
        packedenv = Mock(spec=Env)
        packedenv.action_space = packedspace
        
        component1 = Mock(spec=QAgent)
        component1.agentId = 'a'  # the controlled entity
        component1.getQ = Mock(side_effect=testQCoordinator.maxAtAB16)
        component1.getActionSpace = Mock(return_value=packedspace)

        coordinator = QCoordinator([component1], space, None)
        bestAction = coordinator.step()
        # 16  = 3*5+1
        self.assertEqual(1, bestAction['a'])
        self.assertEqual(5, bestAction['b'])

