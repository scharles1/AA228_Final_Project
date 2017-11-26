import pandas as pd
import numpy as np
from datetime import datetime

startTime = datetime.now()
medium = pd.read_csv('Documents/CS 238/Project 2/medium.csv').as_matrix()

states = 50000
actions = 7
obs = medium.shape[0]
disc = 0.95
alpha = 0.25

#execute Q-learning to estimate utilities of states and actions
Q = np.zeros((states, actions))
for i in range(0, obs):
    s_t = medium[i, 0]
    a_t = medium[i, 1]
    r_t = medium[i, 2]
    sp_t = medium[i, 3]
    Q[s_t - 1, a_t - 1] = Q[s_t - 1, a_t - 1] + alpha * (r_t + disc * max(Q[sp_t - 1, :]) - Q[s_t - 1, a_t - 1])

#iterate through states to return optimal action a for each state
policy_str = ''
counter = 0
for i in range(0, states):
    max_a = np.argmax(Q[i, :]) + 1
    counter = counter + 1
    policy_str = policy_str + '\n' + str(max_a)    
print datetime.now() - startTime
    
f = open('policy_med.policy', 'w')
f.write(policy_str)