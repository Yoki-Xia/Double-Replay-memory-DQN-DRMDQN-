import numpy as np
import matplotlib.pyplot as plt


Q_loss = np.load('Q_loss.npy')
plt.plot(Q_loss)
plt.show()