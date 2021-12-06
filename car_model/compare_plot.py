import numpy as np
import matplotlib.pyplot as plt

def ave_success_rate(data, ave_num):
    success_rate = []
    for i in range(data.shape[0] - ave_num):
        success_rate.append(np.mean(data[i:i+ave_num]))
    return success_rate

def average_success(data):
    success_rate = [int(data[0])]
    # print(success_rate[0])
    for step, i in enumerate(data[1:]):
        result = success_rate[step] + (1/(step+1)) * (i - success_rate[step])
        success_rate.append(result)
    return success_rate

# reward_dqn1 = np.load('成功的训练-回报[-0.1,0.1]/rewards.npy')
# reward_improved_dqn1 = np.load('双经验池训练-01/rewards.npy')
#
# reward_dqn2 = np.load('成功的训练-回报[-0.1,0.1]02/rewards.npy')
# reward_improved_dqn2 = np.load('双经验池训练-02/rewards.npy')
#
# reward_improved_dqn3 = np.load('双经验池训练-03/rewards.npy')
#
# # reward_dqn = ave_success_rate(reward_dqn, 100)
# # reward_improved_dqn = ave_success_rate(reward_improved_dqn, 100)
#
# reward_dqn1 = average_success(data=reward_dqn1)
# reward_improved_dqn1 = average_success(data=reward_improved_dqn1)
#
# reward_dqn2 = average_success(data=reward_dqn2)
# reward_improved_dqn2 = average_success(data=reward_improved_dqn2)
#
# reward_improved_dqn3 = average_success(data=reward_improved_dqn3)
#
# plt.plot(reward_dqn1, c='r', label='dqn_reward')
# plt.plot(reward_improved_dqn1, c='b', label='improved_dqn_reward')
#
# plt.plot(reward_dqn2, c='r', label='dqn_reward')
# plt.plot(reward_improved_dqn2, c='b', label='improved_dqn_reward')
#
# # plt.plot(reward_dqn1, c='r', label='dqn_reward')
# plt.plot(reward_improved_dqn3, c='b', label='improved_dqn_reward')
#
# plt.legend()
# plt.show()

def ave_list(list_2d):
    '''
     输入一个2d的列表，通过平均的方式整合成1d
    约等于将输入的n列数据数据求均值
    '''
    out_p = []
    for i in range(len(list_2d[0])):
        num = 0
        for o in range(len(list_2d)):
            # print(o, i)
            num += list_2d[o][i]
        out_p.append(num/len(list_2d))
    return out_p
def plot_ave_reward():
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'arrives.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'arrives.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('success rate')
    plt.grid()
    # plt.title('success rate')
    plt.ylim((-0.1,1.1))
    plt.legend()
    plt.savefig('map1_success_rate_ave.png')
    plt.show()

def plot_ave_reward2():
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'rewards.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'rewards.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.grid()
    # plt.title('success rate')
    # plt.ylim((-0.1,1.1))
    plt.legend()
    plt.savefig('map1_reward_ave.png')
    plt.show()

def plot_ave_step():
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'train_counts.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'train_counts.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('step')
    plt.grid()
    # plt.title('success rate')
    # plt.ylim((-0.1,1.1))
    plt.legend()
    plt.savefig('map1_step_ave.png')
    plt.show()

def plot_success_rate_reward():
    plt.figure(figsize=(14,4))

    plt.subplot(131)
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'arrives.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'arrives.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment', linestyle='--')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('success rate')
    plt.grid()
    # plt.title('success rate')
    plt.ylim((-0.1, 1.1))
    plt.legend()
    # plt.savefig('map1_success_rate_ave.png')
    # plt.show()
    plt.title('(a) success rate in each episode', y=-0.3)


    plt.subplot(132)
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'rewards.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'rewards.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment',linestyle='--')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.grid()
    # plt.title('success rate')
    # plt.ylim((-0.1,1.1))
    plt.legend()
    plt.title('(b) reward in each episode', y=-0.3)

    plt.subplot(133)
    all = []
    all_2 = []
    for i in range(5):
        file_name = 'single_experiment_pool2/' + str(i) + '/'
        # reward = np.load(file_name + 'success.npy')
        reward = np.load(file_name + 'train_counts.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all.append(reward)
        # plt.plot(reward)
    for i in range(5):
        file_name = 'double_experiment_pool2/' + str(i) + '/'
        reward = np.load(file_name + 'train_counts.npy')
        # reward = average_success(reward)
        reward = ave_success_rate(reward, 500)
        all_2.append(reward)

    ave_reward = ave_list(all)
    # print(len(all_2[0]), len(all_2[1]), len(all_2[2]), len(all_2[3]), len(all_2[4]))
    ave_reward2 = ave_list(all_2)
    plt.plot(ave_reward[0:2000], label='single experiment', linestyle='--')
    plt.plot(ave_reward2[0:2000], label='double experiment')
    plt.xlabel('episode')
    plt.ylabel('step')
    plt.grid()
    # plt.title('success rate')
    # plt.ylim((-0.1,1.1))
    plt.legend()
    plt.title('(c) step in each episode', y=-0.3)

    plt.savefig('map4_all_test.png')
    plt.show()

    # plt.savefig('map1_reward_ave.png')
    # plt.show()

def plot_arrive(file_n):
    for i in range(5):
        file_name = file_n + '/' + str(i) + '/'
        reward = np.load(file_name + 'arrives.npy')
        reward = average_success(reward)
        plt.plot(reward)
    plt.ylim((0,1))
    plt.title(file_n)
    plt.savefig('success rate.png')
    plt.show()


# plot_arrive('single_experiment_pool2')
# plot_ave_reward2()
# plot_ave_step()
plot_success_rate_reward()