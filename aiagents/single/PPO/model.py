import tensorflow as tf
import os
from aiagents.single.PPO.networks import Networks as net
import tensorflow.contrib.layers as c_layers
import numpy as np


class Model(object):

    def __init__(self, parameters, num_actions):
        self.act_size = num_actions
        self.convolutional = parameters['convolutional']
        self.recurrent = parameters['recurrent']
        self.fully_connected = parameters['fully_connected']
        self.influence = parameters['influence']
        self.learning_rate = parameters['learning_rate']
        self.max_step = parameters['max_steps']
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def initialize_graph(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save_graph(self, time_step):
        """
        Retrieve network weights store them.
        """
        with self.graph.as_default():
            file_name = os.path.join('models', self.parameters['name'],
                                     'factor'+str(0), 'network')
            print("Saving networks...")
            self.saver.save(self.sess, file_name, time_step)
            print("Saved!")

    def load_graph(self):
        """
        Load pretrained network weights.
        """
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            checkpoint_dir = os.path.join('models', self.parameters['name'],
                                          'factor'+str(0))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def build_main_model(self):
        """
        Builds neural network
        """
        self.observation = tf.placeholder(shape=[None,
                                                 self.parameters["frame_height"],
                                                 self.parameters["frame_width"],
                                                 self.parameters["num_frames"]],
                                          dtype=tf.float32, name='observation')

        hidden = self.observation
        # normalize input
        if self.parameters['env_type'] == 'atari':
            self.observation_norm = tf.cast(self.observation, tf.float32) / 148.
            hidden = self.observation_norm

        if self.convolutional:
            self.hidden_conv = net.cnn(self.observation,
                                       self.parameters["num_conv_layers"],
                                       self.parameters["num_filters"],
                                       self.parameters["kernel_sizes"],
                                       self.parameters["strides"],
                                       tf.nn.relu, False, 'cnn')
            hidden = c_layers.flatten(self.hidden_conv)

        if self.recurrent:
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32,
                                              name='prev_action')
            self.prev_action_onehot = c_layers.one_hot_encoding(self.prev_action,
                                                                self.act_size)
            hidden = tf.concat([hidden, self.prev_action_onehot], axis=1)

            c_in = tf.placeholder(tf.float32, [None,
                                               self.parameters['num_rec_units']],
                                  name='c_state')
            h_in = tf.placeholder(tf.float32, [None,
                                               self.parameters['num_rec_units']],
                                  name='h_state')
            self.seq_len = tf.placeholder(shape=None, dtype=tf.int32,
                                          name='sequence_length')
            self.state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            hidden, self.state_out = net.rnn(hidden, self.state_in,
                                             self.parameters['num_rec_units'],
                                             self.seq_len,
                                             'rnn')
        if self.fully_connected:
            hidden = net.fcn(hidden, self.parameters["num_fc_layers"],
                             self.parameters["num_fc_units"],
                             tf.nn.relu, 'fcn')
        self.hidden = hidden

    def build_influence_model(self):
        """
        Builds influence model
        """
        def attention(hidden_conv, inf_hidden):
            """
            """
            shape = hidden_conv.get_shape().as_list()
            num_regions = shape[0]*shape[1]
            hidden_conv = tf.reshape(hidden_conv, [num_regions, shape[2]])
            linear_conv = net.fcn(hidden_conv, 1,
                                  self.parameters['num_att_units'], None,
                                  'att', 'att1_{}')
            linear_hidden = net.fcn(inf_hidden, 1,
                                    self.parameters['num_att_units'], None,
                                    'att', 'att2_{}')
            context = tf.nn.tanh(linear_conv + linear_hidden)
            attention_weights = net.fcn(context, 1, [1], None, 'att')
            attention_weights = tf.nn.softmax(attention_weights, axis=0)
            inf_hidden = tf.reduce_sum(attention_weights*hidden_conv, axis=0)
            return [inf_hidden]

        def select_dpatch(hidden_conv):
            """
            """
            inf_hidden = []
            for predictor in range(self.parameters['inf_num_predictors']):
                center = np.array(self.parameters['inf_box_center'][predictor])
                height = self.parameters['inf_box_height'][predictor]
                width = self.parameters['inf_box_width'][predictor]
                predictor_hidden = hidden_conv[center[0]: center[0] + height,
                                               center[1]: center[1] + width, :]
                predictor_hidden = c_layers.flatten(predictor_hidden)
                inf_hidden.append(predictor_hidden)

            inf_hidden = tf.stack(inf_hidden, axis=1)
            hidden_size = inf_hidden.get_shape().as_list()[2]*self.parameters['inf_num_predictors']
            inf_hidden = tf.reshape(inf_hidden, shape=[-1, hidden_size])
            return inf_hidden

        def unroll(iter, state, hidden_states):
            """
            """
            # Retrieve the initial value of each sequence
            s = tf.reshape(tf.cast(iter/self.parameters['inf_seq_len'], tf.int32), [])
            h = tf.reshape(self.inf_state_in.h[s], [1, self.parameters['inf_num_rec_units']])
            c = tf.reshape(self.inf_state_in.c[s], [1, self.parameters['inf_num_rec_units']])
            state0 = tf.contrib.rnn.LSTMStateTuple(c, h)
            # If first element of sequence use intial state else use last state
            state = tf.cond(tf.equal(tf.mod(iter, self.parameters['inf_seq_len']), 0),
                            lambda: state0,
                            lambda: state)
            inf_hidden = state.h
            hidden_conv = self.hidden_conv[iter, :, :, :]
            if self.parameters['attention']:
                inf_hidden = attention(hidden_conv, inf_hidden)
            else:
                inf_hidden = select_dpatch(hidden_conv)
            inf_prev_action_onehot = c_layers.one_hot_encoding(self.inf_prev_action[iter],
                                                               self.act_size)
            inf_hidden = tf.concat([inf_hidden, [inf_prev_action_onehot]], axis=1)
            inf_hidden, state = net.rnn(inf_hidden, state,
                                        self.parameters['inf_num_rec_units'],
                                        self.inf_seq_len, 'inf_rnn')
            hidden_states = hidden_states.write(iter, inf_hidden[0])
            iter += 1
            return [iter, state, hidden_states]

        def condition(iter, state, hidden_states):
            max_iter = tf.shape(self.observation)[0]
            return tf.less(iter, max_iter)

        inf_c = tf.placeholder(tf.float32, [None,
                                            self.parameters['inf_num_rec_units']],
                               name='inf_c_state')
        inf_h = tf.placeholder(tf.float32, [None,
                                            self.parameters['inf_num_rec_units']],
                               name='inf_h_state')
        self.inf_state_in = tf.contrib.rnn.LSTMStateTuple(inf_c, inf_h)
        self.inf_seq_len = tf.placeholder(tf.int32, None,
                                          name='inf_sequence_length')
        self.inf_prev_action = tf.placeholder(shape=[None], dtype=tf.int32,
                                              name='inf_prev_action')
        # outputs of the loop cant change size, thus we need to initialize
        # the hidden states vector and overwrite with new values
        hidden_states = tf.TensorArray(dtype=tf.float32, size=tf.shape(self.observation)[0])
        # Unroll the RNN to fetch intermediate internal states and compute
        # attention weights
        _, self.inf_state_out, hidden_states = tf.while_loop(condition,
                                                             unroll,
                                                             [0,
                                                              self.inf_state_in,
                                                              hidden_states])
        self.inf_hidden = hidden_states.stack()

    def build_optimizer(self):
        """
        """
        raise NotImplementedError

    def evaluate_policy(self):
        """
        """
        raise NotImplementedError

    def evaluate_value(self):
        """
        """
        raise NotImplementedError

    def increment_step(self):
        """
        """
        self.sess.run(self.increment)

    def get_current_step(self):
        """
        """
        return self.sess.run(self.step)
