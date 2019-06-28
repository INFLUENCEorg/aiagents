from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
import logging
from numpy import ndarray


class FactoryFloorAgent(AtomicAgent):
    """
    A Factory Floor Agent that tries to go to the nearest task
    """
    def __init__(self, agentId, environment: FactoryFloor, parameters):
        """
        invert the actions
        """
        super().__init__(agentId, environment, parameters)
        # inverting the key action pairs for meaningful navigation
        self._ACTIONS=dict(zip(environment.ACTIONS.values(), environment.ACTIONS.keys()))

    def step(self, state: FactoryFloorState, reward=None, done=None):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()

        #we need a robot dictionary in the state
        for robot in state.robots:
            if robot._id == self._agentId:
                robotpos=robot.getPosition()
                break

        if not state.tasks:
            return {self._agentId: self._ACTIONS.get("ACT")}

        bestDistance=100
        for task in state.tasks:
            taskpos=task.getPosition()
            distance=sum( abs(robotpos - taskpos) )
            if(distance < bestDistance):
                targetTask = task
                bestDistance = distance
        

        if (targetTask.getPosition() == robot.getPosition()).all():
            action = {self._agentId: self._ACTIONS.get("ACT")}

        if targetTask.getPosition()[0] > robot.getPosition()[0]:
            action = {self._agentId: self._ACTIONS.get("RIGHT")}

        if targetTask.getPosition()[0] < robot.getPosition()[0]:
            action = {self._agentId: self._ACTIONS.get("LEFT")}
       
        if targetTask.getPosition()[1] > robot.getPosition()[1]:
            action = {self._agentId: self._ACTIONS.get("DOWN")}

        if targetTask.getPosition()[1] < robot.getPosition()[1]:
            action = {self._agentId: self._ACTIONS.get("UP")}

        logging.debug(action)
        return action


