import os
import numpy as np
import tensorflow as tf
import time

class DeepQNetwork(object):

    def __init__(self, lr, n_actions, name, fc1_dims=1024,
                 input_dims=(210, 160, 4), chkpt_dir="tmp/dqn"):
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, "deepqnet.ckpt")
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)
        self.write_op = tf.summary.merge([self.accuracy_sum, self.loss_sum, self.summ])
        self.writer = tf.summary.FileWriter("tmp/log_dir")
        self.writer.add_graph(self.sess.graph)


        # The list of values in the collection with the given name
        # or an empty list if no value has been added to that collection.
        # trainable variables are the whose values are updated while performing optimisation.

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims],
                                        name='inputs')
            # * here indicates that the function can take multiple inputs as arguments into the function.
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                         name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                           name='q_target')

            # 1st dimension inside shape is set to None because we want to pass
            # batches of stacked frame into the neural network.

            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
                                     kernel_size=(8, 8), strides=4, name='conv1',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            conv1_activated = tf.nn.relu(conv1)


            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                     kernel_size=(4, 4), strides=2, name='conv2',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            conv2_activated = tf.nn.relu(conv2)


            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128,
                                     kernel_size=(3, 3), strides=1, name='conv3',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            conv3_activated = tf.nn.relu(conv3)

            flat = tf.contrib.layers.flatten(conv3_activated)

            dense1 = tf.layers.dense(flat, units=self.fc1_dims, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))

            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))


            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))
            self.accuracy_sum = tf.summary.scalar('Accuracy', self.q)

            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))
            self.loss_sum = tf.summary.scalar("Loss", self.loss)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            for var in tf.trainable_variables():
                print(var.name[:-2])
                self.summ = tf.summary.histogram(var.name[:-2], var)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('... Saving Checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

class Agent(object):
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=10000, input_dims=(210, 160, 4),
                 q_next_dir="tmp/q_next", q_eval_dir="tmp/q_eval"):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        # for n_actions=3, action_space is a list [0, 1, 2]
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=q_eval_dir)

        # Defining arrays of zeros for state, rewards etc to be stored.
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        # will save a set of 4 stacked frames by number of memories.
        # state_memory has shape(mem_size, 210, 160, 4).

        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        # terminal only has 0 and 1 as input. At end of an episode
        # shape of terminal_memory is (mem_size, 0) with all elements being 0.
        # we don't to have future rewards, this will be indicated by the terminal memory.


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        # actions are arrays of zeros(vector) with len equal to the number of actions. [0, 0, 0]
        # action_space is a list of len equal to the  number of action [0, 1, 2]
        actions[action] = 1.0
        # the action which is given as index is the action chosen.
        # This will assign integer 1 to the action that has ben selected. This is like One hot encoding.
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        # shape (mem_size,)
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal
        # It is a done flag, 0 for done and 1 for not done. It indicates the completion of an episode.
        self.mem_cntr += 1

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            # since action_space has actions as [0, 1, 2], it will generate an integer from elements of action space.
            # That is either 0th, 1st or 2nd action.
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        # we update the graph after every K steps, so that the q_target is not fluctuating.

        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

        batch = np.random.choice(max_mem, self.batch_size)
        # Batch is of the length equal to batch size with elements that are generated using np.arange(max_mem) (b).

        state_batch = self.state_memory[batch]
        # Shape of the state batch is equal to (batch_size, input_dims)
        # ex: (32, 180, 160, 4)

        action_batch = self.action_memory[batch]
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                      feed_dict={self.q_eval.input: state_batch})
        # It has shape (batch_size, n_actions).
        # This gives Q values for each action, in this case 3 actions, using q_eval network for current state batch.

        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                      feed_dict={self.q_next.input: new_state_batch})
        # This gives Q values for the next state using the q_next network.

        q_target = q_eval.copy()
        idx = np.arange(self.batch_size)
        q_target[idx, action_indices] = reward_batch + \
            self.gamma*np.max(q_next, axis=1)*terminal_batch
        # axis= 1 means along each row we calculate the maximum value, where rows are the actions.

        #q_target = np.zeros(self.batch_size)
        #q_target = reward_batch + self.gamma*np.max(q_next, axis =1)*terminal_batch

        _ = self.q_eval.sess.run(self.q_eval.train_op,
                                 feed_dict={self.q_eval.input: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target})

        '''loss = self.q_eval.sess.run(self.q_eval.loss,
                                    feed_dict={self.q_eval.input: state_batch,
                                               self.q_eval.actions: action_batch,
                                               self.q_eval.q_target: q_target})'''



        summary = self.q_eval.sess.run(self.q_eval.write_op,
                                       feed_dict={self.q_eval.input: state_batch,
                                                  self.q_eval.actions: action_batch,
                                                  self.q_eval.q_target: q_target,
                                                  self.q_next.input: new_state_batch})

        self.q_eval.writer.add_summary(summary, time.time())
        self.q_eval.writer.flush()


        if self.mem_cntr > 10000:
            if self.epsilon > 0.05:
                self.epsilon -= 4e-7
            elif self.epsilon <= 0.05:
                self.epsilon = 0.05

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t, e))
