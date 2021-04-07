import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 0, 0.1
x = np.arange(-0.3, 0.3, 0.01)
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
y_inv = sigma * np.sqrt(2 * np.pi) * np.exp((x - mu)**2 / (2 * sigma**2))
plt.plot(x,y)
plt.plot(x,y_inv)
plt.plot(x,y*y_inv)
plt.show()