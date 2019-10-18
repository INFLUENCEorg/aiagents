from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.single.DQN.replay_memory import RandomSampling
from aiagents.single.DQN.DeepQNetwork import DeepQNetwork
import copy
import numpy as np
import random
import logging


class DQNAgent(AtomicAgent):
    def __init__(self, agentId, environment, parameters):
        super().__init__(agentId, environment, parameters)
        # read hyperparameters
        self.epsilon = parameters["epsilon"]
        self.num_frames = parameters["num_frames"]
        self.train_frequency = parameters["train_frequency"]
        # create the replay memory
        self.replay_memory = RandomSampling(
            memory_size = parameters["memory_size"],
            height = parameters["frame_height"],
            width = parameters["frame_width"],
            frames = self.num_frames,
            batch_size = parameters["batch_size"],
        )
        # create placeholders for the current state and previous action
        self.state = None
        self.prev_action = None
        # the number of actions
        self.num_actions = environment.action_space.spaces.get(agentId).n
        # instantiate a deep Q network
        self.deep_q_function = DeepQNetwork(self.num_actions, parameters)
        # step count - used to decide whether to do one training step
        self.step_count = 0

    def get_epsilon_action(self, q_values):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def step(self, observation, reward, done):

        # if done, reset the current state and previous action
        if done is True:
            self.state = None
            self.prev_action=None
            # a placeholder, this action will NOT be executed in the environment
            return {self._agentId: 0}

        # update the current state
        if self.state is not None:
            # update the current state using new observation
            next_state = np.concatenate((self.state[:,:,1:], np.expand_dims(observation, axis=-1)), axis=-1)
            # add the transition into the replay memory
            self.replay_memory.append(self.state, self.prev_action, reward, next_state, done)
            self.state = next_state
        else:
            # initialize the current state at the beginning of an episode
            self.state = np.stack([observation]*self.num_frames, axis=-1)

        # take an epsilon greedy action
        q_values = self.deep_q_function.get_q_values(self.state)[0]
        action = self.get_epsilon_action(q_values)
        self.prev_action = action

        # train deep Q network on one batch of data sampled from replay memory
        if self.replay_memory.full() and self.step_count % self.train_frequency == 0:
            self.deep_q_function.train(self.replay_memory.sample())

        self.step_count += 1

        return {self._agentId: action}
