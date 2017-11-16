import pandas as pd
import numpy as np
from datetime import datetime

small = pd.read_csv('Documents/CS 238/Project 2/small.csv')

states = np.unique(list(small['s']) + list(small['sp']))
num_s = len(states)
actions = np.unique(list(small['a']))
num_a = len(actions)

#calculate transition matrix, rewards
sa_count =  np.zeros((num_s, num_a))
reward = np.zeros((num_s, num_a))
T = np.zeros((num_s, num_s, num_a))
startTime = datetime.now()
for i in range(0, num_s):
    for j in range(0, num_a):
        sa_obs = small.query('s == ' + str(states[i]) + ' & a == ' + str(actions[j]))
        sa_count[i, j] = sa_obs.shape[0]
        reward[i, j] = sum(sa_obs['r']) / sa_count[i, j]
        for k in range(0, num_s):
            T[i, k, j] = sa_obs.query('sp == ' + str(states[k])).shape[0] / sa_count[i, j]
    print i
    print datetime.now() - startTime
    
#finds optimal action and utility
def val_iter(disc):
    error = 0.001  
    conv = False
    u = np.zeros(num_s)
    u_prime = np.zeros(num_s)
    max_a = np.zeros(num_s)
    while conv == False:  
        u = list(u_prime)
        for i in range(0, num_s):
            max_u = -99999999999
            for j in range(0, num_a):
                u_sum = reward[i, j]
                for k in range(0, num_s):
                    u_sum = u_sum + disc * T[i, k, j] * u[k]
                if u_sum > max_u:
                    max_u = u_sum
                    max_a[i] = actions[j]
            u_prime[i] = max_u
            conv = abs(u_prime - u).max() < error
    return u, max_a

val_iter(0.95)
policy = val_iter(0.95)[1]

#converts policy to string for txt file
policy_str = ''
for i in range(0, len(policy)):
    policy_str = policy_str + '\n' + str(policy[i])[:1]