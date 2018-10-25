import numpy as np
import math
import matplotlib.pyplot as plt

data = np.random.normal(4, 1.25, 10000)
plt.hist(data,range=(0,8))
plt.show()
