#state space = outs, runners, count
#reward = - runsscored
#action = all combinations of pitcher handedness, pitchtype, and location

import pandas as pd
import numpy as np

pitches = pd.read_csv('Documents/CS 238/pitches.csv').as_matrix()

def base_code(row):
    first = row[20] > 0.0 #19th column = on_1b
    second = row[21] > 0.0 #20th_column = on_2b
    third = row[22] > 0.0 #21st column = on_3b
    if (first and second and third):
        base = 7
    elif (second and third):
        base = 6
    elif (first and third):
        base = 5
    elif (first and second):
        base = 4
    elif third:
        base = 3
    elif second:
        base = 2
    elif first:
        base = 1
    else:
        base = 0
    return base
    
def get_state(row):
    return 96 * row[7] + 24 * row[8] + 8 * row[9] + base_code(row)

#px -.33, .44
#pz 1.88, 2.65    
def get_action(row):
    if row[16] < -.33:
        q_x = 0
    elif row[16] < .44:
        q_x = 1
    else:
        q_x = 2
    if row[17] < 1.88:
        q_z = 0
    elif row[17] < 2.65:
        q_z = 1
    else:
        q_z = 2
    if row[10] == 'CH':
        p_type = 0
    elif row[10] == 'CU':
        p_type = 1
    elif row[10] == 'FA':
        p_type = 2
    elif row[10] == 'FC':
        p_type = 3
    elif row[10] == 'FS':
        p_type = 4
    elif row[10] == 'SI':
        p_type = 5
    else:
        p_type = 6
    return 9 * p_type + 3 * q_x + q_z
    
rows = pitches.shape[0]
state = np.zeros(rows - 1)
action = np.zeros(rows - 1)
reward = np.zeros(rows - 1)
next_state = np.zeros(rows - 1)

for i in range(0, rows - 1):
    x = pitches[i]
    state[i] = get_state(x)
    action[i] = get_action(x)
    reward[i] = -x[15]
    if x[6] + x[13] == 3:
        next_state[i] = 288
    else:
        y = pitches[i + 1]
        next_state[i] = get_state(y)

obs = np.column_stack((state, action, reward, next_state))
np.savetxt('Documents/CS 238/Final Project/obs.csv', obs, delimiter = ',')