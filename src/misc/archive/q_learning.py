import numpy as np

# Trick to not lose a shape:
# w[:, 0, None]

# This is the discount factor
# what does this do??
gamma = 1

# Number of states
n_states = 3
n_actions = 2

# Value vector. Initialized with zeros.
V = np.zeros(n_states).reshape((1, 1))