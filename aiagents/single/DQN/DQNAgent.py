from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.single.DQN.replay_memory import RandomSampling
from aiagents.single.DQN.DeepQNetwork import DeepQNetwork
import copy
import numpy as np
import random

"""
the idea is to implement DQN in two files
- DQNAgent
- replay_memory (which has already been well implemented)

"""

class DQNAgent(AtomicAgent):
    def __init__(self, agentId, environment, parameters):
        super().__init__(agentId, environment, parameters)
        # create the replay memory
        self.replay_memory = RandomSampling(
            memory_size = parameters["memory_size"],
            height = parameters["frame_height"],
            width = parameters["frame_width"],
            frames = parameters["num_frames"],
            batch_size = parameters["batch_size"],
        )
        self.num_actions = environment.action_space.spaces.get(agentId).n
        self.state = None
        self.prev_action = None
        # initialize the Deep Q Network
        self.deep_q_function = DeepQNetwork(self.num_actions, parameters)

    def step(self, observation, reward, done):
        """
        Jinke Notes:
        * both learning and online decision making happen here
        """

        if done is True:
            self.state = None
            self.prev_action=None
            return {self._agentId: 0} # a placeholder, doesn't matter

        # save transition and update current state given new observation
        if self.state is not None:
            next_state = np.concatenate((self.state[:,:,1:], np.expand_dims(observation, axis=-1)), axis=-1)
            self.replay_memory.append(self.state, self.prev_action, reward, next_state, done)
            self.state = next_state
        else:
            self.state = np.stack([observation]*4, axis=-1)

        # take epsilon-greedy action for the current state
        q_values = self.deep_q_function.get_q_values(self.state)[0]

        if random.random() < 0.1:
            action = random.randint(0, self.num_actions-1)
        else:
            action = np.argmax(q_values)

        # TODO: train frequency not done yet
        if self.replay_memory.full():
            self.deep_q_function.train(self.replay_memory.sample())

        self.prev_action = action

        return {self._agentId: action}
