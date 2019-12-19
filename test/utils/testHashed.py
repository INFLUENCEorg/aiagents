

from test.LoggedTestCase import LoggedTestCase
from aienvs.Environment import Env
from unittest.mock import Mock
from gym.spaces import  Discrete
from aiagents.QAgentComponent import QAgentComponent
from aiagents.utils.Hashed import Hashed


class testHashed(LoggedTestCase):

    def test_smoke(self):
        Hashed(1)

    def test_hashcode(self):
        hash(Hashed(1))
        hash(Hashed({'a':1}))
        hash(Hashed(Discrete(2)))

    def test_eq(self):
        self.assertEqual(Hashed(1), Hashed(1))
        self.assertEqual(Hashed({'a':1}), Hashed({'a':1}))
        self.assertEqual(Hashed(Discrete(2)), Hashed(Discrete(2)))

    def test_hashcode_eq(self):
        self.assertEqual(hash(Hashed(1)), hash(Hashed(1)))
        self.assertEqual(hash(Hashed({'a':1})), hash(Hashed({'a':1})))
        self.assertEqual(hash(Hashed(Discrete(2))), hash(Hashed(Discrete(2))))
    
