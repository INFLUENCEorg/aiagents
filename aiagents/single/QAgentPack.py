from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.QAgentComponent import QAgentComponent
from aienvs.Environment import Env
from gym import spaces
from math import exp
import random
from aiagents.utils.Hashed import Hashed
from aienvs.gym.DecoratedSpace import DecoratedSpace
from aienvs.gym import PackedSpace


class QAgentPack(QAgentComponent):
    """
    An adapter that takes a QAgentComponent that was working on 
    a packed space and turns it into a QAgentComponent that 
    works on a non-packed space. Needed for operations on unpacked
    spaces like QCoordinator.
    """

    def __init__(self, agt: QAgentComponent, packing: PackedSpace):
        self._agent = agt
        self._packedspace = packing

    # Override
    def getQ(self, state, action:dict) -> float:
        return self._agent.getQ(state, self._packedspace.pack action)
    
    # Override
    def getV(self, state):
        return None  # what should this do anyway?
    
