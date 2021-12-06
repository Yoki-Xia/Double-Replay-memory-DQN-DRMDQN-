import numpy as np

actions = np.random.randint(0,3,(5000 ))
print(actions)
np.save('actions.npy',actions)