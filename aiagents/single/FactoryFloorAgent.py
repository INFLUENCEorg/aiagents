from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
import logging
import random
from numpy import ndarray, hypot


class FactoryFloorAgent(AtomicAgent):
    """
    A Factory Floor Agent that tries to go to the nearest task
    """

    def step(self, state: FactoryFloorState, reward=None, done=None):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()
        action = random.randint(0, self._environment.action_space.spaces.get(self._agentId).n-1)

#we need a robot dictionary in the state
        for robot in state.robots:
            if robot._id == self._agentId:
                robotpos=robot.getPosition()

        for task in state.tasks:
            taskpos=task.getPosition()
            distance=sum( abs(robotpos - taskpos) )

        actions.update({self._agentId: action })
        logging.debug("Id / action:" + str(actions))
        return actions

