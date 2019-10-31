

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym import spaces
from gym.spaces import  Dict
from aiagents.QAgentComponent import QAgentComponent
from aiagents.single.QAgent import QAgent


class testQAgent(LoggedTestCase):

    def test_smoke(self):
        QAgent("agent1", Mock())
