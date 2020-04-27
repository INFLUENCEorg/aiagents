import numpy as np
import random
import os

class ReplayMemory(object):
    """
    Database to store transitions in.
    """
    def __init__(self, memory_size, height, width, frames, batch_size):
        """
        Initialize the memory with the right size.
        """
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.states = np.zeros((memory_size, height, width, frames),dtype="float32")
        self.actions = np.zeros((memory_size,), dtype="int32")
        self.next_states = np.zeros((memory_size, height, width, frames),dtype="float32")
        self.terminal_states = np.zeros((memory_size,), dtype="int32")
        self.rewards = np.zeros((memory_size, ), dtype="float32")

        self.pointer = 0
        self.items = 0

    def append(self, state, action, reward, next_state, terminal):
        """
        Add the given information to the Replay Memory.
        """
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.terminal_states[self.pointer] = int(terminal)

        self.pointer += 1
        self.items += 1
        if self.pointer == self.memory_size:
            # Restart from 0
            self.pointer = 0

    def sample(self, batch_size):
        """
        Sample a batch from the dataset. This can be implemented in
        different ways.
        """
        raise NotImplementedError

    def full(self):
        """
        Check whether the replay memory has been filled.
        """
        return self.items >= self.memory_size

    @property
    def fullness(self):
        """
        Return the percentage the replay memory has been filled.
        """
        return min(float(self.items)/self.memory_size, 1.0)*100

    def store(self, path):
        """
        Store the necessary information to recreate the replay memory.
        """
        # We combine the states and the next_states, as they overlap in
        # information. States goes from s_0 to s_T-1 and next_states goes from
        # s_1 to s_T.
        outfile = os.path.join(path, "replay_memory.npz")
        all_states = np.vstack((self.states, [self.next_states[-1]]))
        np.savez(outfile, states=all_states, actions=self.actions, rewards=self.rewards, terminals=self.terminal_states)

    def load(self, path):
        """
        Load a stored replay memory.
        """
        # Go two levels up
        outfile = os.path.join(path, "replay_memory.npz")
        npzfile = np.load(outfile)
        self.states = npzfile['states'][:-1]
        self.next_states = npzfile['states'][1:]
        self.actions = npzfile['actions']
        self.rewards = npzfile['rewards']
        self.terminal_states = npzfile['terminals']
        self.items = self.memory_size

class RandomSampling(ReplayMemory):
    """
    Database to sample random batches from. (does not sample sequences)
    """
    def __init__(self, memory_size, height, width, frames, batch_size):
        """
        This is an instance of a plain Replay Memory object. It does not
        need more information than its super class.
        """
        ReplayMemory.__init__(self, memory_size, height, width, frames, batch_size)

    def sample(self):
        """
        Sample a batch from the dataset by chosing the transitions at
        random.
        """
        indices = np.arange(self.memory_size)
        random.shuffle(indices)

        selected_states = self.states[indices[:self.batch_size]]
        selected_actions = self.actions[indices[:self.batch_size]]
        selected_rewards = self.rewards[indices[:self.batch_size]]
        selected_next_states = self.next_states[indices[:self.batch_size]]

        return selected_states, selected_actions, selected_rewards, selected_next_states
