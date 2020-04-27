from aiagents.single.PPO.PPO import PPO
from aiagents.single.PPO.buffer import Buffer
import numpy as np
from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
import logging
import tensorflow as tf
import os
from gym.spaces import Dict


class PPOAgent(AtomicAgent):

    def __init__(self, agentId, actionspace:Dict, observationspace, parameters:dict):
        AtomicAgent.__init__(self, agentId, actionspace, observationspace, parameters)
        self._prev_state = None
        # TODO: change to self._step_output = dict({"obs": observation_space.sample(), "reward": None, "done": None, "prev_action": None})
        self._step_output = None
        self._action = [-1]
        self._parameters = parameters
        self._num_actions = actionspace.n
        self._train_frequency = self._parameters['train_frequency']
        self._save_frequency = self._parameters['save_frequency']
        self._agentId = agentId
        self._PPO = PPO(self._parameters, self._num_actions)
        self._buffer = Buffer(self._parameters, self._num_actions)
        self._cumulative_rewards = 0
        self._episode_step = 0
        self._episodes = 1
        self._t = 0
        self._stats = {"cumulative_rewards": [],
                        "episode_length": [],
                        "value": [],
                        "learning_rate": [],
                        "entropy": [],
                        "policy_loss": [],
                        "value_loss": []}
        tf.reset_default_graph()
        self._step = 0
        summary_path = 'summaries/' + self._parameters['name'] + '_' + \
                        self._parameters['algorithm']
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self._summary_writer = tf.summary.FileWriter(summary_path)
        if self._parameters['influence']:
            self._seq_len = self._parameters['inf_seq_len']
        elif self._parameters['recurrent']:
            self._seq_len = self._parameters['seq_len']
        else:
            self._seq_len = 1

    def step(self, obs, reward, done):
        # new_state = np.zeros((1, self._parameters['frame_height'],
        #                       self._parameters['frame_width'],
        #                       self._parameters['num_frames']))
        # frame = observation
        # prev_state = np.copy(new_state)
        # stacked_obs[0, :, :, 0] = frame
        # new_state[0, :, :, 1:] = prev_state[:, :, :, :-1]
        if self._parameters['obs_type'] == 'image':
            obs = np.stack(np.split(obs,2))
            obs = [np.swapaxes(obs, 0, 2)]

        else:
            obs = [obs]
        next_step_output = dict({"obs": obs, "reward": reward, "done": done,
                                  "prev_action": self._action})
        if self._step_output is None:
            self._step_output = next_step_output
        else:
            if self._parameters['mode'] == 'train':
                # Store experiences in buffer
                self._add_to_memory(self._step_output,
                                    next_step_output,
                                    self._action_output)
                # Calculate advantages and returns
                self._bootstrap(next_step_output)
                if self._step % self._train_frequency == 0 and self._full_memory():
                    self._update()
                if self._step % self._save_frequency == 0:
                    # Tensorflow only stores a limited number of networks.
                    self._save_graph(self._step)
                self._write_summary()
            self._step_output = next_step_output
        if done:
            self._step_output = None
            self._action = [-1]
            return {self._agentId: self._action}

        self._action_output = self._get_action(self._step_output)
        self._increment_step()
        self._action = self._action_output.get('action')
        return {self._agentId: self._action}

    ############# PRIVATE METHODS ####################

    def _add_to_memory(self, step_output, next_step_output, get_actions_output):
        """
        Append the last transition to buffer and to stats.
        """
        self._buffer['obs'].append(step_output['obs'][0])
        self._buffer['rewards'].append(next_step_output['reward'])
        self._buffer['dones'].append(next_step_output['done'])
        self._buffer['actions'].append(get_actions_output['action'])
        self._buffer['values'].append(get_actions_output['value'])
        self._buffer['action_probs'].append(get_actions_output['action_probs'])
        # NO ZERO-PADDING ANYMORE SEQUENCES MIGHT CONTAIN EXPERIENCES FROM
        # TWO DIFFERENT EPISODES. SEE MASKS BELOW.
        # This mask is added so we can ignore experiences added when
        # zero-padding incomplete sequences
        self._buffer['masks'].append(1)
        self._cumulative_rewards += next_step_output['reward']
        logging.debug("Cumulative reward" + str(self._cumulative_rewards) + " reward" + str(next_step_output['reward']))
        self._episode_step += 1
        self._stats['value'].append(get_actions_output['value'])
        self._stats['entropy'].append(get_actions_output['entropy'])
        self._stats['learning_rate'].append(get_actions_output['learning_rate'])
        # Note: States out is used when updating the network to feed the
        # initial state of a sequence. In PPO this internal state will not
        # differ that much from the current one. However for DQN we might
        # rather set the initial state as zeros like in Jinke's
        # implementation
        if self._parameters['recurrent']:
            self._buffer['states_in'].append(
                    np.transpose(get_actions_output['state_in'], (1, 0, 2)))
            self._buffer['prev_actions'].append(step_output['prev_action'])
        if self._parameters['influence']:
            self._buffer['inf_states_in'].append(
                    np.transpose(get_actions_output['inf_state_in'], (1, 0, 2)))
            self._buffer['inf_prev_actions'].append(step_output['prev_action'])
        if next_step_output['done']:
            self._stats['cumulative_rewards'].append(self._cumulative_rewards)
            self._stats['episode_length'].append(self._episode_step)
            self._cumulative_rewards = 0
            self._episode_step = 0
        if self._parameters['recurrent'] or self._parameters['influence']:
            if next_step_output['done']:
                # reset worker's internal state
                self._PPO.reset_state_in()
                # NOTE: FIND OUT HOW TO RESTART THE RNN'S INTERNAL STATE
                # WHEN UPDATING THE MODEL USING SEQUENCES THAT CORRESPOND
                # TO TWO DIFFERENT EPISODES
                self._buffer['masks'].append(0)
            else:
                self._buffer['masks'].append(1)

    def _bootstrap(self, next_step_output):
        """
        Computes GAE and returns for a given time horizon
        """
        self._t += 1
        if self._t >= self._parameters['time_horizon']:
            evaluate_value_output = self._PPO.evaluate_value(
                                        next_step_output['obs'],
                                        next_step_output['prev_action'])
            next_value = evaluate_value_output['value']
            batch = self._buffer.get_last_entries(self._t, ['rewards', 'values',
                                                            'dones'])
            advantages = self._compute_advantages(np.array(batch['rewards']),
                                                 np.array(batch['values']),
                                                 np.array(batch['dones']),
                                                 next_value,
                                                 self._parameters['gamma'],
                                                 self._parameters['lambda'])
            self._buffer['advantages'].extend(advantages)
            returns = advantages + batch['values']
            self._buffer['returns'].extend(returns)
            self._t = 0

    def _update(self):
        """
        Runs multiple epoch of mini-batch gradient descent to update the model
        using experiences stored in buffer.
        """
        policy_loss = 0
        value_loss = 0
        n_sequences = self._parameters['batch_size'] // self._seq_len
        n_batches = self._parameters['memory_size'] // \
            self._parameters['batch_size']
        for e in range(self._parameters['num_epoch']):
            self._buffer.shuffle()
            for b in range(n_batches):
                batch = self._buffer.sample(b, n_sequences)
                update_model_output = self._PPO.update_model(batch)
                policy_loss += update_model_output['policy_loss']
                value_loss += update_model_output['value_loss']
        self._buffer.empty()
        self._stats['policy_loss'].append(np.mean(policy_loss))
        self._stats['value_loss'].append(np.mean(value_loss))

    def _compute_advantages(self, rewards, values, dones, last_value, gamma,
                            lambd):
        """
        Calculates advantages using genralized advantage estimation (GAE)
        """
        last_advantage = 0
        advantages = np.zeros((self._parameters['time_horizon'], 1),
                              dtype=np.float32)
        for t in reversed(range(self._parameters['time_horizon'])):
            mask = 1.0 - dones[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[t] + gamma * last_value - values[t]
            last_advantage = delta + gamma * lambd * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]
        return advantages

    def _get_action(self, step_output, ip=None):
        """
        Get each factor's action based on its local observation. Append the given
        state to the factor's replay memory.
        """
        evaluate_policy_output = {}
        evaluate_policy_output.update(self._PPO.evaluate_policy(step_output['obs'],
                                                                step_output['prev_action']))
        return evaluate_policy_output

    def _full_memory(self):
        """
        Check if the replay memories are filled.
        """
        return self._buffer.full()

    def _save_graph(self, time_step):
        """
        Store all the networks and replay memories.
        """
        # Create factor path if it does not exist.
        factor_path = os.path.join('models', self._parameters['name'])
        if not os.path.exists(factor_path):
            os.makedirs(factor_path)
        self._PPO.save_graph(time_step)

    def _increment_step(self):
        self._PPO.increment_step()
        self._step = self._PPO.get_current_step()

    def _write_summary(self):
        """
        Saves training statistics to Tensorboard.
        """
        if self._step % self._parameters['summary_frequency'] == 0 and \
           self._parameters['tensorboard']:

            summary = tf.Summary()
            for key in self._stats.keys():
                if len(self._stats[key]) > 0:
                    stat_mean = float(np.mean(self._stats[key]))
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self._stats[key] = []
            self._summary_writer.add_summary(summary, self._step)
            self._summary_writer.flush()
