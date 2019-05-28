from aiagents.single.PPO.controller import Controller
from aiagents.single.PPO.PPOmodel import PPOmodel
from aiagents.single.PPO.replay_memory import SerialSampling
import numpy as np
from aiagents.AgentComponent import AgentComponent
import logging


class PPOAgent(Controller, AgentComponent):

    def observe(self, observation, reward, done):
        new_state = np.zeros((1, self._parameters['frame_height'],
                              self._parameters['frame_width'],
                              self._parameters['num_frames']))
        frame = observation
        prev_state = np.copy(new_state)
        new_state[0, :, :, 0] = frame
        new_state[0, :, :, 1:] = prev_state[:, :, :, :-1]

        next_step_output = dict({"obs": new_state, "reward": reward, "done": done, "prev_action": self._prev_action})
        if( self._step_output["obs"] is None ):
            self._step_output = next_step_output

        self.increment_step()
        get_actions_output = self._get_actions(self._step_output)

        if self._parameters['mode'] == 'train':
            # Store the transition in the replay memory.
            self._add_to_memory(self._step_output,
                                next_step_output,
                                get_actions_output, self._t)
            self._bootstrap(next_step_output)
            if self._step % self._train_frequency == 0 and self._full_memory():
                self._update()
            if self._step % self._save_frequency == 0:
                # Tensorflow only stores a limited number of networks.
                self._save_graph(self._step)
            self.write_summary()

        self._step_output = next_step_output

    def select_actions(self):
        action=self._get_actions(self._step_output)
        self._prev_action=action.get('action')[0]
        return {self._controller_id: action['action'][0][0]}

    def __init__(self, parameters, observation_space, action_space, controller_id=0, logger=None):
        self._num_actions = {}
        self._prev_state = None
        self._observation_space = observation_space
        #TODO: change to self._step_output = dict({"obs": observation_space.sample(), "reward": None, "done": None, "prev_action": None})
        self._step_output = dict({"obs": None, "reward": None, "done": None, "prev_action": None})
        self._prev_action = [-1]
        self._parameters = parameters
        self._num_actions = action_space.n
        self._train_frequency = self._parameters['train_frequency']
        self._save_frequency = self._parameters['save_frequency']
        self._controller_id = controller_id
        self._model = {}
        self._replay_memory = {}
        self._cumulative_rewards = {}
        self._stats = {}
        self._model = PPOmodel(self._parameters, self._num_actions)
        self._replay_memory = SerialSampling(self._parameters, self._num_actions)
        self._cumulative_rewards = 0
        self._stats = {"cumulative_rewards": [],
                        "episode_length": [],
                        "value": [],
                        "learning_rate": [],
                        "entropy": [],
                        "policy_loss": [],
                        "value_loss": []}
        super().__init__(self._parameters, action_space)
        self._t = 0
        if self._parameters['influence']:
            self._seq_len = self._parameters['inf_seq_len']
        elif self._parameters['recurrent']:
            self._seq_len = self._parameters['seq_len']
        else:
            self._seq_len = 1


    ############# PRIVATE METHODS ####################

    def _add_to_memory(self, step_output, next_step_output, get_actions_output,
                      episode_step):
        """
        Append the last transition to the replay memories of the factors.
        """
        # The given actions come from the experimentor and are actions,
        # but we need action indices.
        # TODO: this does not work as expected with factors. We would like to
        # append the values of each factor to the corresponding key in the dictionary
        self._replay_memory['obs'].append(step_output['obs'])
        self._replay_memory['rewards'].append(next_step_output['reward'])
        self._replay_memory['dones'].append(next_step_output['done'])
        self._replay_memory['actions'].append(get_actions_output['action'])
        self._replay_memory['values'].append(get_actions_output['value'][0])
        self._replay_memory['action_probs'].append(get_actions_output['action_probs'])
        # This mask is added so we can ignore experiences added when zero
        # padding incomplete sequences
        self._replay_memory['masks'].append(1)
        self._cumulative_rewards += next_step_output['reward']
        logging.debug("Cumulative reward"+str(self._cumulative_rewards))
        self._stats['value'].append(get_actions_output['value'][0])
        self._stats['entropy'].append(get_actions_output['entropy'])
        self._stats['learning_rate'].append(get_actions_output['learning_rate'])
        # Note: States out is used when updating the network to feed the
        # initial state of a sequence. In PPO this internal state will not
        # differ that much from the current one. However for DQN we might
        # rather set the initial state as zeros like in Jinke's
        # implementation
        if self._parameters['recurrent']:
            self._replay_memory['states_in'].append(get_actions_output['state_in'])
            self._replay_memory['prev_actions'].append(step_output['prev_action'])
        if self._parameters['influence']:
            self._replay_memory['inf_states_in'].append(get_actions_output['inf_state_in'])
            self._replay_memory['inf_prev_actions'].append(step_output['prev_action'])

        if next_step_output['done']:
            if self._parameters['recurrent'] or self._parameters['influence']:
                self._model.reset_state_in()
            self._stats['cumulative_rewards'].append(self._cumulative_rewards)
            self._stats['episode_length'].append(episode_step)
            self._cumulative_rewards = 0
            # zero padding incomplete sequences
            remainder = len(self._replay_memory['masks']) % self._seq_len
            if remainder != 0:
                missing = self._seq_len - remainder
                self._replay_memory.zero_padding(missing)
                self._t += missing

    def _bootstrap(self, next_step_output):
        """
        """
        # TODO: consider the case where the episode is over because the maximum
        # number of steps in an episode has been reached.
        self._t += 1
        if self._t >= self._parameters['time_horizon']:
            evaluate_value_output = self._model.evaluate_value(
                                        next_step_output['obs'],
                                        next_step_output['prev_action'])
            next_value = evaluate_value_output['value']
            batch = self._replay_memory.get_last_entries(self._t, ['rewards', 'values', 'dones'])
            advantages = self._compute_advantages(np.reshape(batch['rewards'],
                                                            (1, -1)),
                                                 np.reshape(batch['values'],
                                                            (1, -1)),
                                                 np.reshape(batch['dones'],
                                                            (1, -1)),
                                                 next_value,
                                                 self._parameters['gamma'],
                                                 self._parameters['lambda'])
            self._replay_memory['advantages'].extend(advantages)
            returns = advantages + np.reshape(batch['values'],-1)
            self._replay_memory['returns'].extend(returns)
            self._t = 0

    def _update(self):
        """
        Sample a batch from the replay memory (if it is completely filled) and
        use it to update the models.
        """
        import time
        start = time.time()
        logging.debug(len(self._replay_memory['returns']))
        logging.debug(len(self._replay_memory['masks']))
        policy_loss = 0
        value_loss = 0
        n_sequences = self._parameters['batch_size'] // self._seq_len
        n_batches = self._parameters['memory_size'] // self._parameters['batch_size']
        for e in range(self._parameters['num_epoch']):
            self._replay_memory.shuffle()
            for b in range(n_batches):
                batch = self._replay_memory.sample(b, n_sequences)
                update_model_output = self._model.update_model(batch)
                policy_loss += update_model_output['policy_loss']
                value_loss += update_model_output['value_loss']
        self._replay_memory.empty()
        self._stats['policy_loss'].append(np.mean(policy_loss))
        self._stats['value_loss'].append(np.mean(value_loss))
        end = time.time()
        logging.debug(end - start)

    def _compute_advantages(self, rewards, values, dones, last_value, gamma, lambd):
        """
        """
        last_advantage = 0
        advantages = np.zeros(self._parameters['time_horizon'], dtype=np.float32)
        for t in reversed(range(self._parameters['time_horizon'])):
            mask = 1.0 - dones[:, t]
            last_value = last_value*mask
            last_advantage = last_advantage*mask
            delta = rewards[:, t] + gamma*last_value - values[:, t]
            last_advantage = delta + gamma*lambd*last_advantage
            advantages[t] = last_advantage
            last_value = values[:, t]
        return advantages
