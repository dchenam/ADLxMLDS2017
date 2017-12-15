from agent_dir.agent import Agent
import gym
import os
import numpy as np
import pickle
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class Policy:
    def __init__(self, learning_rate, decay_rate, gamma, n_actions):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []

        with tf.name_scope('inputs'):
            self.observations = tf.placeholder(tf.float32, [None, 6400])
            self.sampled_actions = tf.placeholder(tf.int32, [None])
            self.advantage = tf.placeholder(tf.float32, [None])

        fc1 = tf.layers.dense(
            self.observations,
            units=200,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(6400)))

        logits = tf.layers.dense(
            fc1,
            units=n_actions,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(200))
        )

        self.prob = tf.nn.softmax(logits)
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.sampled_actions, logits=logits)
            weighted_log_prob = tf.multiply(neg_log_prob, self.advantage)
            loss = tf.reduce_mean(weighted_log_prob)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(loss)

    def sample(self, sess, observations):
        prob = sess.run(
            self.prob,
            feed_dict={self.observations: np.reshape(observations, [1, -1])})
        action = np.random.choice(3, p=prob.ravel())
        return action

    def store_transition(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def discount_rewards(self):
        R = 0
        discount_rewards = []
        # Discount Rewards
        for r in self.rewards[::-1]:
            if r != 0: R = 0  # reset the sum, after someone scores a point
            R = r + self.gamma * R
            discount_rewards.insert(0, R)
        # Normalize Rewards
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + np.finfo(np.float32).eps
        return discount_rewards

    def train(self, sess):
        discount_reward = self.discount_rewards()
        feed_dict = {
            self.observations: np.vstack(self.states),
            self.sampled_actions: np.array(self.actions),
            self.advantage: discount_reward
        }
        sess.run(self.train_op, feed_dict)
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]

def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

class Agent_PG(Agent):
    def __init__(self, env, args):
        super(Agent_PG,self).__init__(env)
        # hyperparameters
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.decay_rate = 0.99
        self.n_actions = 3

        #Game space
        self.env = env
        self.i_episode = 0
        self.prev_state = None
        self.running_reward = None
        self.reward_history = []

        self.sess = tf.InteractiveSession()
        self.policy = Policy(learning_rate=self.learning_rate, decay_rate = self.decay_rate,
                             gamma=self.gamma, n_actions=self.n_actions)
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.experiment_dir = os.path.abspath("./saved/REINFORCE")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.checkpoint_file = os.path.join(self.checkpoints_dir,
                                            'policy_network.ckpt')

        if args.test_pg:
            self.load_checkpoint()

    def init_game_setting(self):
        pass

    def train(self):
        time_step = 0
        reward_sum = 0

        while True:
            #Reset Episode Parameters
            done = False
            reward_epsiode = 0
            round_n = 1
            state = self.env.reset()

            while not done:
                #Compute State Delta
                state = prepro(state)
                # state_delta = state - self.prev_state if self.prev_state is not None else state
                # self.prev_state = state

                #Sample Stochastic Policy
                action = self.policy.sample(self.sess, state)
                # Step Environment and Update Reward
                next_state, reward, done, _ = self.env.step(action + 1)

                # Record Game History
                self.policy.store_transition(state, action, reward)
                reward_epsiode += reward
                round_n += 1
                time_step += 1
                state = next_state

            self.policy.train(self.sess)
            reward_sum += reward_epsiode

            if self.i_episode % 30 == 0:
                average_reward = reward_sum / 30
                reward_sum = 0
                self.reward_history.append([time_step, average_reward])

            # Logging
            self.running_reward = reward_epsiode if self.running_reward is None else self.running_reward * 0.99 + reward_epsiode * 0.01

            if self.i_episode % 10 == 0:
                print(
                    'ep {}: reward: {}, mean reward: {:3f}'.format(self.i_episode, reward_epsiode, self.running_reward))
            else:
                print('\tep {}: finished after {} rounds, reward: {}'.format(self.i_episode, round_n, reward_epsiode))

            # Save model in every 50 episode
            if self.i_episode % 50 == 0:
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
        self.i_episode = int(latest_checkpoint.split('-')[-1])

    def save_checkpoint(self):
        print("Saving checkpoint...")
        if not os.path.exists('./tf_checkpoints'):
            os.makedirs('./tf_checkpoints')
        self.saver.save(self.sess, self.checkpoint_file, global_step=self.i_episode)
        pickle.dump(self.reward_history, open('./tf_checkpoints/reward_history.p', 'wb'))

    def make_action(self, observation, test=True):
        # Preprocess State
        state = prepro(observation)
        # Sample Stochastic Policy
        action = self.policy.sample(self.sess, state)
        return action + 1

