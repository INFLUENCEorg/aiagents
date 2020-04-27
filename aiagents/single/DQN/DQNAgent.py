from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.listener.DefaultListenable import DefaultListenable
from aienvs.listener.Listener import Listener
from .replay_memory import RandomSampling
from .DeepQNetwork import DeepQNetwork
from gym.spaces import Dict
import copy
import numpy as np
import random
import logging

class DQNAgent(AtomicAgent, DefaultListenable, Listener):
    """
    DQN Agent. 

    The idea is to make this as minimal as possible.
    """
    def __init__(self, agentId, actionspace:Dict=None, observationspace=None, parameters:dict=None):
        super().__init__(agentId, actionspace, observationspace, parameters)

        # observation
        parameters['frame_height'] = observationspace.shape[0]
        parameters['frame_width'] = observationspace.shape[1]

        # get the number of actions
        self.num_actions = actionspace.n
        print("Number of actions for agent {}: {}".format(agentId, self.num_actions))

        # create placeholders for the current state and previous action
        self.state = None
        self.prev_action = None

        # load hyperparameters
        self.epsilon = parameters["epsilon"]
        self.num_frames = parameters["num_frames"]
        self.train_frequency = parameters["train_frequency"]
        
        # instantiate a replay memory
        self.replay_memory = RandomSampling(
            memory_size = parameters["memory_size"],
            height = parameters['frame_height'],
            width = parameters['frame_width'],
            frames = self.num_frames,
            batch_size = parameters["batch_size"],
        )
        
        # instantiate a deep Q network
        self.deep_q_function = DeepQNetwork(self.num_actions, parameters)

        # count how many environmental steps the agent has experienced
        # used to decide when to perform a training step
        self.step_count = 0

        # in eval mode, the agent always takes actions greedily according to the value function
        # in training mode, the agent uses epsilon greedy policy to guarantee sufficient exploration
        self._eval = False

    def eval(self):
        self._eval = True
    
    def train(self):
        self._eval = False

    def get_epsilon_greedy_action(self, q_values):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def get_greedy_action(self, q_values):
        return np.argmax(q_values)

    def step(self, observation, reward, done):
        
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

        # train deep Q network on one batch of data sampled from replay memory
        if self._eval is False and self.replay_memory.full() and self.step_count % self.train_frequency == 0:
            self.deep_q_function.train(self.replay_memory.sample())

        # if done, reset the current state and previous action
        if done is True:
            self.reset()
            return {self._agentId: 0} # this action does not matter -> will not be actually taken

        # select an action
        q_values = self.deep_q_function.get_q_values(self.state)[0]
        action = None
        if self._eval is True:
            action = self.get_greedy_action(q_values)
        else:
            action = self.get_epsilon_greedy_action(q_values)
        self.prev_action = action

        self.step_count += 1

        return {self._agentId: action}

    def reset(self):
        self.state = None
        self.prev_action = None

    def getQ(self, state, action):
        q_values = self.deep_q_function.get_q_values(state)[action] 
        return q_values

    def getV(self, state):
        q_values = self.deep_q_function.get_q_values(state)
        v_values = np.max(q_values, axis=-1)
        return v_values
    
    def notifyChange(self, data):
        self.notifyAll(data)
