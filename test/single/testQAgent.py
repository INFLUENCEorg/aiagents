

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym import spaces
from gym.spaces import  Dict
from aiagents.QAgentComponent import QAgentComponent
from aiagents.single.QAgent import QAgent
from aiagents.utils.Hashed import Hashed

STATE1 = Hashed('state1')
STATE2 = Hashed('state2')
STATE3 = Hashed('state3')
ACT1 = Hashed('action1')
ACT2 = Hashed('action2')


class testQAgent(LoggedTestCase):

    def test_smoke(self):
        QAgent("agent1", Mock())
        
    def test_getQ(self):
        agent = QAgent("agent1", Mock())
        self.assertEqual(0, agent._getQ(STATE1, ACT1))

    def test_updateQ_step1(self):
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':1, 'm':500, 's':0.01})
        agent._updateQ(STATE2, ACT2, STATE3, 2)
        self.assertEqual(0.2, agent._getQ(STATE2, ACT2))
    
    def test_updateQ_step2(self):
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':.9, 'm':500, 's':0.01})
        agent._updateQ(STATE2, ACT2, STATE3, 2)
        agent._updateQ(STATE1, ACT1, STATE2, 3)
        self.assertEqual(0.1 * (3 + .9 * 0.2), agent._getQ(STATE1, ACT1))

    def test_updateQ_step11(self):
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':1, 'm':500, 's':0.01})
        agent._updateQ(STATE2, ACT2, STATE3, 2)
        agent._updateQ(STATE2, ACT2, STATE3, 2)
        self.assertEqual(0.9 * 0.2 + 0.1 * 2, agent._getQ(STATE2, ACT2))

    def test_getMaxQ(self):
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':1, 'm':500, 's':0.01})
        agent._updateQ(STATE1, ACT1, STATE2, 2)
        agent._updateQ(STATE1, ACT2, STATE3, 3)
        self.assertEqual(0.1 * 3, agent._getMaxQ(STATE1))
        self.assertEqual(ACT2, agent._getMaxAction(STATE1))
    
    def test_chooseActionBest(self):
        # -500,1 results in p=1 everywhere so always choose BEST action
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':1, 'm':-500, 's':1})
        agent._updateQ(STATE1, ACT1, STATE2, 2)
        agent._updateQ(STATE1, ACT2, STATE3, 3)
        for i in range(10):
            # chooseaction may act randomly. test multiple times.
            self.assertEqual(ACT2, agent._chooseAction(STATE1))
        
    def test_chooseActionRandom(self):
        # 500,1 results in p=0 everywhere so always choose RANDOM action
        agent = QAgent("agent1", Mock(), {'alpha':0.1, 'gamma':1, 'm':500, 's':1})
        agent._updateQ(STATE1, ACT1, STATE2, 2)
        agent._updateQ(STATE1, ACT2, STATE3, 3)
        chosen = [agent._chooseAction(STATE1) for x in range(100)] 
        
        
