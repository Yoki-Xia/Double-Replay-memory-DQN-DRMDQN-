import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import time

class DQNreplayer():            #经验池
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity        #决定经验池的容量

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count+1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)




class agent_DQN():
    def __init__(self, env, gamma=0.95, learning_rate=0.001, epsilon=0.01,
                 batch_size=512, capacity=150000, load_model=False, hidden_layers=[32,32,32,32,32], give_weight_num=10):
        self.action_n = env.action_n
        # self.observation_n = env.observation_n
        self.map_size = env.map_size
        self.gamma = gamma          #折扣因子
        self.learning_rate = learning_rate      #神经网络的学习率
        self.epsilon = epsilon          #贪心策略的随机概率
        self.replayer = DQNreplayer(capacity=capacity)
        # self.replayer_minus = DQNreplayer(capacity=capacity)
        self.batch_size = batch_size
        self.give_weight_num = give_weight_num
        self.give_weight_count = 0.0
        self.learn_interval = 0.
        self.net_input = 18         # 神经网络输入参数的数目
        self.Q_loss = []

        if load_model:
            self.target_net = my_model(lazer_num=15, output_size=self.action_n, learning_rate=learning_rate)
            self.target_net.load_weights('target_net.ckpt')
            self.evaluate_net = my_model(lazer_num=15, output_size=self.action_n, learning_rate=learning_rate)
            self.evaluate_net.load_weights('evaluate_net.ckpt')

            self.replayer = pd.read_pickle('memory.pickle')
            # self.replayer_minus = pd.read_pickle('memory_minus.pickle')
            # m = len(self.replayer['observation'])
            #
            # self.replayer.columns = m-1
            # self.replayer.i = m-1
            # print('load_memory_success')
            # print(self.replayer.memory)
            # input('pause!')
            print('加载目标网络和评估网络成功')
        else:
            self.evaluate_net = my_model(lazer_num=15, output_size=self.action_n, learning_rate=learning_rate)
            self.target_net = my_model(lazer_num=15, output_size=self.action_n, learning_rate=learning_rate)

            self.target_net.set_weights(self.evaluate_net.get_weights())
            print('创建目标网络和评估网络成功')

    def load_observation(self, observation_str):
        '''
        读取observation，因为之前储存的时候变成了string类型的数据，需要转化一下
        :param observation_str:
        :return:
        '''
        # print(observation_str)
        observation_str = observation_str.lstrip('[')
        observation_str = observation_str.rstrip(']')
        l = np.array([float(i) for i in observation_str.split()])
        # print(l)
        # input('p')
        return l

    def decide(self, observation):
        '''
        动作决策
        :param observation:
        :return:
        '''
        # self.epsilon = 10000/(10000 + self.replayer.capacity)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0,self.action_n)
        else:
            # print(observation[np.newaxis].shape)
            q = self.evaluate_net.call(observation[np.newaxis])
            return np.argmax(q)

    def learn(self, observation, action, reward, next_observation, done,
              to_learn, learning_interval0):
        self.replayer.store(observation, action, reward, next_observation, done)
        # if reward > 0:
        #     self.replayer.store(observation, action, reward, next_observation, done)
        # else:
        #     self.replayer_minus.store(observation, action, reward, next_observation, done)
        if to_learn:
            self.learn_interval += 1
            # print('get learn')
            if self.learn_interval >= learning_interval0:
                self.learn_interval = 0
                # print('learning!')
                observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
                # observations_minus, actions_minus, rewards_minus, next_observations_minus, dones_minus = self.replayer_minus.sample(self.batch_size)
                # # 合并来自不同经验池中提取的数据
                # observations = np.vstack([observations, observations_minus])
                # actions = np.hstack([actions, actions_minus])
                # rewards = np.hstack([rewards, rewards_minus])
                # next_observations = np.vstack([next_observations, next_observations_minus])
                # dones = np.hstack([dones, dones_minus])

                # print('observation :',observations.shape)
                # print('actions :', actions.shape)
                # print('rewards :', rewards.shape)
                # print('next_observations :', next_observations.shape)
                # print('dones :', dones.shape)
                # input('ppp')

                next_qs = self.target_net(next_observations)
                qs = self.evaluate_net(observations).numpy()
                # qs = qs.numpy()
                # qs[0, 0] = 1
                # print(qs.shape, qs)
                # qs[0,0] = 1
                next_max_qs = np.max(next_qs, axis=1)
                # print(next_max_qs.shape)
                # print('observation : ',observations.shape)
                # input('pause')
                for i in range(len(rewards)):
                    # print(qs[i,actions[i]], rewards[i], int(dones[i]), next_max_qs[i])
                    qs[i, actions[i]] = rewards[i] + self.gamma * (1 - int(dones[i])) * next_max_qs[i]

                self.Q_loss.append(self.evaluate_net.fit(observations, qs))


                self.give_weight_count += 1
                if self.give_weight_count >= self.give_weight_num:
                    self.target_net.set_weights(self.evaluate_net.get_weights())
                    self.give_weight_count = 0.
                    print('网络更新！evaluate_net -> target_net')

''' 建立神经网络结构 '''
def build_net(input_size, output_size=4, hidden_layers=[],learning_rate=0.01,
              hidden_activation=tf.nn.relu, output_activation=None):
    '''
    构建神经网络
    :param input_size:
    :param output_size:
    :param hidden_layer:
    :param learning_rate:
    :param hidden_activation:
    :param output_activation:
    :return:
    '''
    model = tf.keras.Sequential()
    for hidden_layer in hidden_layers:
        model.add(layers.Dense(hidden_layer, activation=hidden_activation))
    model.add(layers.Dense(output_size,activation=output_activation))
    model.build(input_shape=(None, input_size))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.losses.MSE
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model

class my_model(tf.keras.Model):
    def __init__(self, lazer_num, output_size, learning_rate):
        super(my_model, self).__init__()
        self.conv1d_1 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_2 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_3 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_4 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.d1 = layers.Dense(32, activation='relu')
        self.d2 = layers.Dense(32, activation='relu')
        self.d3 = layers.Dense(32, activation='relu')
        self.d4 = layers.Dense(output_size, activation=None)
        self.lazer_num = lazer_num
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.conv1d_3 = layers.Conv1D(1, 3, strides=2, activation='relu')
        # self.conv1d_4 = layers.Conv1D(16, 3, activation='relu')
        # self.conv1d_5 = layers.Conv1D(32, 3, activation='relu')
        # self.conv1d_6 = layers.Conv1D(64, 3, activation='relu')

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        # lazer = x[:,0:self.lazer_num,:]
        lazer = x[:, 0:, :]
        # others = x[:,self.lazer_num:,:]
        # 处理激光数据
        # lazer = self.conv1d_1(lazer)
        # lazer = self.conv1d_2(lazer)
        # lazer = self.conv1d_3(lazer)
        # lazer = self.conv1d_4(lazer)
        # 数据类型转换
        lazer = tf.cast(lazer, dtype=tf.float32)
        # others = tf.cast(others, dtype=tf.float32)
        # 数据类型结合继续计算
        # total = tf.concat([lazer, others], axis=1)
        # print(total.shape)
        if len(lazer.shape) == 3:
            total = tf.squeeze(lazer, axis=2)
        # print('total 1 : ', total.shape)
        total = self.d1(total)
        # print('total 2 : ', total.shape)
        total = self.d2(total)
        # print('total 3 : ', total.shape)
        total = self.d3(total)
        # print('total 4 : ', total.shape)
        total = self.d4(total)
        # print('total 5 : ', total.shape)
        return total

    def fit(self, x, y):
        with tf.GradientTape() as tape:
            pre_y = self.call(x)
            loss = tf.keras.losses.mean_squared_error(pre_y, y)
            # print('loss shape : ', loss.shape)
            loss_sum = tf.reduce_mean(loss)
            # print('loss_sum shape : ', loss_sum.shape)
        gradients = tape.gradient(loss_sum, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_sum


def play_DNQ(env, agent, rander=False, training=True, sleep=False, learning_interval=4):
    episode_reward = 0
    # 走一步初始化智能体中的激光传感器
    env.step(2)
    observation = env.get_observations()
    training_count = 0.
    while True:
        if rander:
            env.render()
        if agent.train_step > 0 and training:
            action = np.random.randint(0, env.action_n)
            action = agent.actions[agent.train_step-1]
            # print(action)
            agent.train_step -= 1
        else:
            action = agent.decide(observation)
            # print('action : ', action)
            # input('pause !!!')
        reward, done, arrive = env.step(action)

        next_observation = env.get_observations()
        episode_reward += reward
        # print('data : ', observation*env.map_size, action, reward, next_observation*env.map_size, done)
        # print(observation, observation.shape)
        # input('暂停查看数据 ： ')
        # print(observation, training_count)
        # if observation.shape[0] == 18:
        if agent.train_step <= 0:
            if training:
                agent.learn(observation, action, reward, next_observation, done,
                            to_learn=True, learning_interval0=learning_interval)
        else:
            if training:
                agent.learn(observation, action, reward, next_observation, done,
                            to_learn=False, learning_interval0=learning_interval)

        training_count += 1
        if sleep:
            time.sleep(0.3)
        if done:
            env.reset(arrive)
            break
        if training_count == 2000:
            env.reset(arrive)
            break
        observation = next_observation.copy()
    return episode_reward, training_count, arrive



# def get_observations(env):
#     '''
#     智能体获取深度学习中的参数
#     :param env:
#     :return:
#     '''
#     agent_pos = np.array(env.agt1_pos)
#     goal_pos = np.array(env.goal_pos)
#     observation = np.hstack([agent_pos, goal_pos]) / env.map_size
#     # print(observation)
#     return observation
