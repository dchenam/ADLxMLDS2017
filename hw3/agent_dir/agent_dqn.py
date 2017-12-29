from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import pickle
import os
from collections import namedtuple
from itertools import count
import random
import sys

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, experiment_dir):
        self.n_actions = 4
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.gamma = 0.99
        self.decay_rate = 0.99

        self.summary_writer = None
        summary_dir = os.path.join(experiment_dir, "summaries")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.states = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.actions = tf.placeholder(tf.int32, [None])
        self.rewards = tf.placeholder(tf.float32, [None])
        self.states_ = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.dones = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('eval_net'):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                self.states, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.contrib.keras.layers.LeakyReLU(alpha=0.01))
            self.q_eval = tf.contrib.layers.fully_connected(
                fc1, self.n_actions)

        with tf.variable_scope('target_net'):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                self.states_, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.contrib.keras.layers.LeakyReLU(alpha=0.01))
            self.q_next = tf.contrib.layers.fully_connected(
                fc1, self.n_actions)

        #Maybe Update Target Network
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_update = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        q_target = self.rewards + self.gamma * self.dones * tf.reduce_max(self.q_next, axis=1)
        self.q_target = tf.stop_gradient(q_target) #Freeze Target Network

        # Get the predictions for the chosen actions only
        self.q_action = tf.reduce_sum(self.q_eval * tf.one_hot(self.actions, depth=self.n_actions), axis=1)

        # Calculate q values and targets
        # Loss = E[(r + gamma * max(Q target) - Q(eval))^2]
        self.loss = tf.reduce_max(tf.squared_difference(self.q_target, self.q_action, name='td_error'))
        #self.loss = tf.losses.huber_loss(self.q_target, self.q_action)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.decay_rate)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = self.optimizer.compute_gradients(self.loss, var_list=self.e_params)
        # for i, (grad, var) in enumerate(gradients):
        #     if grad is not None:
        #         gradients[i] = (tf.clip_by_norm(grad, self.clip_val), var)
        self.train_op = self.optimizer.apply_gradients(gradients,
                                                       global_step=tf.train.get_global_step())
        # self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.q_eval))
        ])

    def choose_action(self, sess, observation, epsilon):
        # Epsilon Based Exploration
        observation = np.expand_dims(observation, 0)
        if np.random.uniform() > epsilon:
            q_values = sess.run(self.q_eval, feed_dict={self.states: observation})
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def train(self, sess, replay_memory):
        # Sample a minibatch from the replay memory
        batch_memory = replay_memory.sample(self.batch_size)

        # Perform gradient descent update
        summaries, global_step,  _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict={
                self.states: batch_memory.state,
                self.actions: batch_memory.action,
                self.rewards: batch_memory.reward,
                self.states_: batch_memory.next_state,
                self.dones: np.invert(batch_memory.done).astype(np.float32)
            }
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        # Game space
        self.env = env
        self.num_steps = 1e7
        self.num_episodes = 100000
        self.i_episode = 0
        self.policy_update_freq = 4
        self.target_update_freq = 1000
        self.reward_history = []
        self.valid_actions = [0, 1, 2, 3]
        self.initial_memory_size = 10000
        self.memory_size = 10000
        self.replay_memory = ReplayMemory(self.memory_size)

        # The epsilon decay schedule
        epsilon_start = 1.0,
        epsilon_end = 0.05,
        self.epsilon_decay_steps = 1000000
        self.epsilons = np.linspace(epsilon_start, epsilon_end, self.epsilon_decay_steps)

        self.experiment_dir = os.path.abspath("./saved/Vanilla_DQN")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        tf.reset_default_graph()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.sess = tf.InteractiveSession()
        self.dqn = DQN(self.experiment_dir)
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.checkpoints_dir,
                                            'DQN_network.ckpt')
        self.total_t = self.sess.run(tf.train.get_global_step())

        if args.test_dqn:
            self.load_checkpoint()

    def init_game_setting(self):

        pass


    def train(self):
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes),
            episode_rewards=np.zeros(self.num_episodes))
        reward_sum = 0
        print("Populating replay memory...")
        state = self.env.reset()
        for i in range(self.initial_memory_size):
            action = self.dqn.choose_action(self.sess, state, self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)])
            next_state, reward, done, _ = self.env.step(action)
            self.replay_memory.push(state, action, reward, next_state, done)
            self.total_t += 1
            if done:
                state = self.env.reset()
            else:
                state = next_state
        running_reward = None
        while self.total_t < self.num_steps:
            # Reset Episode Parameters
            reward_epsiode = 0
            state = self.env.reset()
            loss = None
            for t in count(1):
                epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
                if self.total_t % self.target_update_freq == 0:
                    self.sess.run(self.dqn.target_update)
                    print('\ntarget network updated')

                # print("\rStep {} ({}) @ Episode {}/{}, reward: {}".format(
                #     t, self.total_t, self.i_episode + 1, self.num_episodes, reward), end="")
                # sys.stdout.flush()

                action = self.dqn.choose_action(self.sess, state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                # Save Transition to Memory
                self.replay_memory.push(state, action, reward, next_state, done)

                # Update statistics
                stats.episode_rewards[self.i_episode] += reward
                stats.episode_lengths[self.i_episode] = t

                if self.total_t % self.policy_update_freq == 0:
                    self.dqn.train(self.sess, self.replay_memory)

                reward_epsiode += reward
                state = next_state
                self.total_t += 1
                if done:
                    break
            running_reward = reward_epsiode if running_reward is None else running_reward * 0.99 + reward_epsiode * 0.01
            print("\rSteps {} @ Episode {}/{}, reward_episode: {}, running_reward: {:3f}".format(
                self.total_t, self.i_episode, self.num_episodes, reward_epsiode, running_reward), end="")
            sys.stdout.flush()

            if self.i_episode % 30 == 0:
                average_reward = reward_sum / 30
                reward_sum = 0
                step_summary = tf.Summary()
                step_summary.value.add(simple_value=average_reward, tag="30_episode/reward")
                self.dqn.summary_writer.add_summary(step_summary, self.total_t)
                self.reward_history.append([self.total_t, average_reward])

            # Logging
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
            episode_summary.value.add(simple_value=stats.episode_rewards[self.i_episode], tag="episode/reward")
            episode_summary.value.add(simple_value=stats.episode_lengths[self.i_episode], tag="episode/length")
            self.dqn.summary_writer.add_summary(episode_summary, self.i_episode)
            self.dqn.summary_writer.flush()

            reward_sum += reward_epsiode

            # Save model in every 50 episode
            if self.i_episode % 500 == 0:
                print('ep %d: model saving...' % (self.i_episode))
                self.save_checkpoint()

            self.i_episode += 1

    def load_checkpoint(self):
        print("Loading checkpoint...")
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoints_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        self.reward_history = pickle.load(open(os.path.join(self.experiment_dir, 'reward_history.p'), 'rb'))

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file, global_step=self.i_episode)
        pickle.dump(self.reward_history, open(os.path.join(self.experiment_dir, 'reward_history.p'), 'wb'))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        action = self.dqn.choose_action(self.sess, observation, 0.05)
        return action

