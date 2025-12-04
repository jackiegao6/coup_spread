import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

tran_exponent_a = 2.0
n = 10000

tran = np.random.power(tran_exponent_a, n)

plt.hist(tran, bins=50, density=True)
plt.title(f"Power Distribution (a={tran_exponent_a})")
plt.show()
