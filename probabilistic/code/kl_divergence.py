import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
sns.set()

"""
define kl-divergence function 
KL(P||Q) = Sum(Pi(x).log(Pi(x)/Qi(x))) if P and Q distribution are discreate random variable 
KL (P||Q) =  integral(p(x)log(p(x)/q(x))dx)
"""

"""
don’t include any probabilities equal to 0 because the log of 0 is negative infinity
"""


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# toys data x
x = np.arange(-10, 10, 0.001)
p = norm.pdf(x, 0, 2)
q = norm.pdf(x, 2, 2)

plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q))
plt.plot(x, p)
plt.plot(x, q, c='red')

plt.show()
