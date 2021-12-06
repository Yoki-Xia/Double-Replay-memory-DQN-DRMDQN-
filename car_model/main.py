import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import bicycle_model
import subfunction as sf
import scipy
import tensorflow as tf

# print(tf.)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

env = bicycle_model.KinematicModel(psi=0, v=5, f_len=0.5, r_len=0.5)

# 手动控制智能体
while True:
        total_reward, step_count, arrive = bicycle_model.play_env(env)
        print(total_reward, step_count, arrive)

''' 智能体参数设置 '''
show_result = False            #是否查看运行结果
give_weight_num = 5           #学习多少次以后给目标网络赋值

''' 训练情况设置 '''
agent1 = sf.agent_DQN(env, gamma=0.95, learning_rate=0.001 , epsilon=0.01, give_weight_num=give_weight_num,load_model=False)
render = False
stop = False
train= True
sleep = False
agent1.train_step = 5000     # 刚开始训练的时候多少次后开始训练
agent1.actions = np.load('actions.npy')

''' DQN 参数设置'''
episode = 1500           # 运行次数
rewards = []            # 储存回报
# agnet1.train_step = 5000     # 刚开始训练的时候多少次后开始训练
learning_interval = 8     # 走多少步学习一次
train_counts = []
train_at_begin = False
arrives = []

for i in range(episode):
    reward, train_count, arrive = sf.play_DNQ(env, agent1, render, train, sleep=sleep, learning_interval=learning_interval)
    # if train_step > 0:
    #     train_step -= train_count
    #     # if train_step <= 0:
    #         # render = True
    #     print(train_step)
    if agent1.train_step > 0:
        print(agent1.train_step)

    print('第 {} 回合的回报为 {}, 执行步数为 {}, 是否到达目标处 {} '.format(i+1, reward, train_count, arrive))

    rewards.append(reward)
    train_counts.append(train_count)
    arrives.append(arrive)

    #每十次训练就保存一下数据，这样有利于实时观察数据情况
    if (i+1) % 100 == 0:
        np.save('rewards.npy', rewards)
        np.save('train_counts.npy', train_counts)
        np.save('arrives.npy', arrives)
        np.save('Q_loss.npy', np.array(agent1.Q_loss))

    if (i+1) % 100 == 0:
        agent1.target_net.save_weights('target_net.ckpt')
        agent1.evaluate_net.save_weights('evaluate_net.ckpt')
        agent1.replayer.memory.to_pickle('memory.pickle')
        agent1.replayer_minus.memory.to_pickle('memory_minus.pickle')
        print('-------------saved!----------------')

plt.plot(rewards)
plt.title('path planning')
plt.xlabel('training num')
plt.ylabel('reward')
plt.savefig('reward.png')
plt.show()

plt.plot(train_counts)
plt.title('path planning')
plt.xlabel('training num')
plt.ylabel('step')
plt.savefig('step.png')
plt.show()

plt.plot(arrives)
plt.title('path planning')
plt.xlabel('training num')
plt.ylabel('arrive')
plt.savefig('arrive.png')
plt.show()

np.save('rewards.npy', rewards)
np.save('train_counts.npy', train_counts)
np.save('arrives.npy', arrives)