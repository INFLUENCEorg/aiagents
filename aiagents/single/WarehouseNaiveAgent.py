from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
from aienvs.gym.DecoratedSpace import DecoratedSpace
import logging
import networkx as nx
import random
import numpy as np

class WarehouseNaiveAgent(AtomicAgent):
    """
    WarehouseNaiveAgent moves towards the closest item in its domain.
    """
    ACTIONS = {'UP': 0,
               'DOWN': 1,
               'LEFT': 2,
               'RIGHT': 3}
    def __init__(self, robot_id:str, env: Env, parameters:dict=None):
        super().__init__(robot_id, env, parameters)
        full_action_space = DecoratedSpace.create(env.action_space)
        self._action_space=full_action_space.get(robot_id)
        self._env = env
        self._robot_id = robot_id
        self._robot_domain = self._env.robots[self._robot_id].get_domain
        self._graph = self._env.create_graph(self._env.robots[self._robot_id])
        self._action_mapping = {(-1, 0): self.ACTIONS.get('UP'),
                                (1, 0): self.ACTIONS.get('DOWN'),
                                (0, -1): self.ACTIONS.get('LEFT'),
                                (0, 1): self.ACTIONS.get('RIGHT')}
        # Compute shortest paths between all nodes
        self._path_dict = dict(nx.all_pairs_dijkstra_path(self._graph))

    def step(self, state, reward=None, done=None):
        """
        Make one step towards the closest item
        """
        path = self._path_to_closest_item()
        if path is None or len(path) < 2:
            action = random.randint(0, self._action_space.getSize()-1)
        else:
            action = self._get_first_action(path)
        action = {self._robot_id: action}
        logging.debug("Id / action:" + str(action))

        return action

    def _path_to_closest_item(self):
        """
        Calculates the distance of every item in the robot's domain, finds the
        closest item and returns the path to that item.
        """
        min_distance = self._robot_domain[2] - self._robot_domain[0] + \
                       self._robot_domain[3] - self._robot_domain[1]
        closest_item_path = None
        for item in self._env.items:
            if self._robot_domain[0] <= item.get_position[0] <= self._robot_domain[2] and \
              self._robot_domain[1] <= item.get_position[1] <= self._robot_domain[3]:
                path = self._path_dict[tuple(self._env.robots[self._robot_id].get_position)][tuple(item.get_position)]
                distance = len(path) - 1
                if distance < min_distance:
                    min_distance = distance
                    closest_item_path = path
        return closest_item_path

    def _get_first_action(self, path):
        """
        Get first action to take in a given path
        """
        delta = tuple(np.array(path[1]) - np.array(path[0]))
        action = self._action_mapping.get(delta)
        return action
