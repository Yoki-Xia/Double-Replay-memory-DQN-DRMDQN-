import numpy as np
import matplotlib.pyplot as plt

success = np.load('arrives.npy')

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
    return  success_rate



success_rate = ave_success_rate(success, 100)
# success_rate = average_success(success)
plt.plot(success_rate)
plt.xlabel('training num')
plt.ylabel('success rate')
plt.ylim((0, 1))
plt.grid()
plt.savefig('success_rate.png')
plt.show()

np.save('success_rate.npy', success_rate)
