import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import bicycle_model
import subfunction as sf


env = bicycle_model.KinematicModel(psi=0, v=5, f_len=0.5, r_len=0.5)

# 手动控制智能体
# while True:
#         total_reward, step_count, arrive = bicycle_model.play_env(env)
#         print(total_reward, step_count, arrive)

'''智能体参数设置'''
show_result = False            #是否查看运行结果
give_weight_num = 5           #学习多少次以后给目标网络赋值

if show_result:
    '''查看训练完成的 agent1 '''
    agent1 = sf.agent_DQN(env, epsilon=0., load_model=True)
    render = True
    stop = True
    train = False
    sleep = True

else:
    '''训练使用'''
    agent1 = sf.agent_DQN(env, gamma=0.95, learning_rate=0.001 , epsilon=0.01, give_weight_num=give_weight_num,load_model=True)
    render = True
    stop = False
    train= True
    sleep = False

''' DQN 参数设置'''
episode = 500           # 运行次数
rewards = []            # 储存回报
train_step = 0      # 刚开始训练的时候多少次后开始训练
learning_interval = 10     # 走多少步学习一次
train_counts = []
train_at_begin = False
arrives = []

for i in range(episode):
    reward, train_count, arrive = sf.play_DNQ(env, agent1, train_step, render, train, sleep=sleep, learning_interval=learning_interval)
    if train_step > 0:
        train_step -= train_count
        # if train_step <= 0:
            # render = True
        print(train_step)

    print('第 {} 回合的回报为 {}, 执行步数为 {}, 是否到达目标处 {} '.format(i+1, reward, train_count, arrive))

    rewards.append(reward)
    train_counts.append(train_count)
    arrives.append(arrive)
    if (i+1) % 100 == 0:
        agent1.target_net.save('target_net.h5')
        agent1.evaluate_net.save('evaluate_net.h5')
        agent1.replayer.memory.to_csv('memory.csv')
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