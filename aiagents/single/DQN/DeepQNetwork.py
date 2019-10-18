import tensorflow as tf
from .q_function import Q_function

class DeepQNetwork(Q_function):
    """
    Deep Q-Networks have a lot in common. This class describes the
    functionalities of such a network.
    """
    def __init__(self, num_actions, parameters):
        """
        This is an instance of a Q-network.
        """
        Q_function.__init__(self, num_actions, parameters)
        """
        Create Tensorflow variables
        """
        with self.graph.as_default():
            with tf.variable_scope('input'):
                # These variables represent the variables that go into the network(s)
                self.states = tf.placeholder(
                    tf.float32,
                    [None, parameters["frame_height"], parameters["frame_width"], self.num_frames],
                    name='states'
                )
                self.next_states = tf.placeholder(
                    tf.float32,
                    [None, parameters["frame_height"], parameters["frame_width"],
                    self.num_frames], name='next_states'
                )
                # These variables are used to calculate the loss
                self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')
                self.actions = tf.placeholder(tf.int32, [None, ], name='actions')
        # Create the action network
        self.q_vals, act_net = self.inference(self.states, 'action')

        # This creates a new network graph (e.g. it creates the target network)
        self.next_q_vals, targ_net = self.inference(self.next_states, 'target')

        self.sync_networks = self.reset_q_hat(act_net, targ_net)

        # Calculate the loss given the calculated components.
        if parameters["double_dqn"]:
            double_q_vals, _ = self.inference(self.next_states, 'action', True)
        else:
            double_q_vals = None
        loss, self.td = self.loss_function(self.q_vals, self.next_q_vals, self.rewards, self.actions, double_q_vals)

        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                # Only update the variables of the action network
                self.opt_operation = self.updates.minimize(loss, var_list=list(act_net.values()))

            self.sess.run(tf.global_variables_initializer())
        print("Finished building network.")

    def get_q_values(self, state):
        """
        Compute the Q-values for the given state.

        Keyword arguments:
            state -- the state to compute Q-values for (1, width, height,
            num_frames) nparray
        """
        q_vals = self.sess.run(self.q_vals, feed_dict={self.states: [state],
                                                       self.batch_size: 1})
        return q_vals

    def train(self, batch):
        """
        Train one batch.

        Keyword arguments:
        states -- mini batch of states
        actions -- mini batch of actions
        rewards -- mini batch of rewards
        next_states -- mini batch of next states

        Returns: average squared loss
        """
        states, actions, rewards, next_states = batch

        if self.freeze_interval > 0 and self.update_counter % self.freeze_interval == 0:
            self.sess.run(self.sync_networks)
            print("Set target network to action network.")
        if self.update_counter % self.lra == 0 and self.update_counter > 0:
            self.lr = self.lr * self.ar
            print("Annealed learning rate on step {} to {}".format(self.update_counter, self.lr))

        var_dict = {self.states: states, self.next_states: next_states, self.actions: actions,
                    self.rewards: rewards, self.learning_rate: self.lr, self.batch_size: self.b_s}

        # if self.update_counter % self.summary_frequency == 0:
        #     summary, _ = self.sess.run([self.summaries, self.opt_operation], feed_dict=var_dict)
        #     self.train_writer.add_summary(summary, self.update_counter)
        # else:
        
        # no summary at the moment
        self.sess.run(self.opt_operation, feed_dict=var_dict)
        self.update_counter += 1

    def td_error(self, batch):
        """
        Computes the TD error for the given batch of transitions.

        Keyword arguments:
        states -- mini batch of states
        actions -- mini batch of actions
        rewards -- mini batch of rewards
        next_states -- mini batch of next states

        Returns: a vector
        """
        states, actions, rewards, next_states = batch

        # Calculate action network outputs
        diff = self.sess.run(self.td, feed_dict={self.states: states, self.next_states: next_states,
                                                 self.actions: actions, self.rewards: rewards, self.batch_size: 1})
        return diff

    def inference(self, x, name, doube_dqn=False):
        """
        This is the forward pass through the network given an input state.
        Args:
            x:      4D float Tensor of size [batch_size, height, width, num_features]
            name:   String containing the name of this network (e.g. action or target)
        Returns:
            logits: 2D float Tensor of size [batch_size, num_actions]
            vars:   a dictionary containing the variables of this network
        """
        # We store all variables in this dictionary
        var = {}
        # We don't want to include the regularization losses of the target network
        if name == 'action':
            reg = tf.contrib.layers.l2_regularizer(self.weight_decay)
        else:
            reg = None

        with self.graph.as_default():
            with tf.variable_scope(name, reuse=doube_dqn):
                with tf.variable_scope('conv1'):
                    var['conv1_w'] = tf.get_variable('filter', (8, 8, self.num_frames, 32), initializer=self.init_func,
                                                     regularizer=reg)
                    var['conv1_b'] = tf.get_variable('biases', (32,), initializer=tf.constant_initializer(0.1))
                    # Do the calculations of this layer
                    h = tf.nn.conv2d(x, var['conv1_w'], strides=[1, 4, 4, 1], padding='VALID')
                    h = tf.nn.bias_add(h, var['conv1_b'])
                    h = tf.nn.relu(h, name='activation')
                # with tf.variable_scope('conv2'):
                #     var['conv2_w'] = tf.get_variable('filter', (4, 4, 32, 64), initializer=self.init_func,
                #                                      regularizer=reg)
                #     var['conv2_b'] = tf.get_variable('biases', (64,), initializer=tf.constant_initializer(0.1))
                #     # Do the calculations of this layer
                #     h = tf.nn.conv2d(h, var['conv2_w'], strides=[1, 2, 2, 1], padding='VALID')
                #     h = tf.nn.bias_add(h, var['conv2_b'])
                #     h = tf.nn.relu(h, name='activation')
                # with tf.variable_scope('conv3'):
                #     var['conv3_w'] = tf.get_variable('filter', (3, 3, 64, 64), initializer=self.init_func,
                #                                      regularizer=reg)
                #     var['conv3_b'] = tf.get_variable('biases', (64,), initializer=tf.constant_initializer(0.1))
                #     # Do the calculations of this layer
                #     h = tf.nn.conv2d(h, var['conv3_w'], strides=[1, 1, 1, 1], padding='VALID')
                #     h = tf.nn.bias_add(h, var['conv3_b'])
                #     h = tf.nn.relu(h, name='activation')
                with tf.variable_scope('flatten'):
                    # Reshape the output of the conv layers to be flat
                    n_input = h.get_shape().as_list()[1] * h.get_shape().as_list()[2] * h.get_shape().as_list()[3]
                    h = tf.reshape(h, [-1, n_input])
                with tf.variable_scope('lin1'):
                    var['lin1_w'] = tf.get_variable('weights', (n_input, 512), initializer=self.init_func,
                                                    regularizer=reg)
                    var['lin1_b'] = tf.get_variable('biases', (512,), initializer=tf.constant_initializer(0.1))
                    h = tf.matmul(h, var['lin1_w']) + var['lin1_b']
                    h = tf.nn.relu(h, name="activation")
                with tf.variable_scope('lin2'):
                    var['lin2_w'] = tf.get_variable('weights', (512, self.num_actions), initializer=self.init_func,
                                                    regularizer=reg)
                    var['lin2_b'] = tf.get_variable('biases', (self.num_actions,),
                                                    initializer=tf.constant_initializer(0.1))
                    logits = tf.matmul(h, var['lin2_w']) + var['lin2_b']

        return logits, var
