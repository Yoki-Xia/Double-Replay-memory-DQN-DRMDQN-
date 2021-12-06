import numpy as np
import matplotlib.pyplot as plt

rewards = np.load('rewards.npy')
train_counts = np.load('train_counts.npy')
arrives = np.load('arrives.npy')

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

# np.save('rewards.npy', rewards)
# np.save('train_counts.npy', train_counts)
# np.save('arrives.npy', arrives)