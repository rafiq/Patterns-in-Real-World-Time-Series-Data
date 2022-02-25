import numpy as np


# lets say that you found complexity values
K = np.array([3.2, 5.6, 3.2, 7.8, 1.5])

# we scale the complexity in this way
N = 4#number of unique different patterns found
K_scaled = np.log2(N)*( K - np.min(K) )/( np.max(K) - np.min(K))

# the one issue is that I am not sure whether to set
# N = number of unique different patterns found
# or
# N = maximum number of strings of the given length, eg if 5 bit strings then N=2**5=32. Let's do both and see which is best.

# As for the Upper bound, it is just

Up_Bound = 2**-K_scaled

# you just need to add that on to your plots