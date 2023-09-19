from matplotlib import pyplot as plt
import numpy as np


arr = [[1,2,3], [4,5,6], [7,8,9]]

N_blocks = len(arr)
N_chains = len(arr[0])
for i in range(N_chains):
    chains = np.array(arr)[:,i]
    iter = np.linspace(0, N_blocks-1, N_blocks)
    plt.scatter(iter, chains, label="Chain %s" % i)
plt.show()