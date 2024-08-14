import csv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

CSV_FILE_NAME_01 = 'csv_data_/test_pr_0.1_reward.csv'
CSV_FILE_NAME_02 = 'csv_data_/test_pr_0.2_reward.csv'

prob01matrix = np.loadtxt(CSV_FILE_NAME_01, delimiter=",", dtype = float,skiprows=1)
prob02matrix = np.loadtxt(CSV_FILE_NAME_02, delimiter=",", dtype = float,skiprows=1)




'''timestep_list = []
reward_list = []

with open(CSV_FILE_NAME_02, mode ='r')as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        if lines[0] == "Timestep":
            continue
        timestep_list.append(float(lines[0]))
        reward_list.append(float(lines[1]))'''


#print(timestep_list)
#print(reward_list)

# plotting ----------------------------------------------------------------------------------------

#fig, ax1 = plt.subplots()

#mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])

plt.plot(prob02matrix[:,0], prob01matrix[:,1], color = '#ec5f59', label = 'car prob. = 0.1') # prob 0.1
plt.plot(prob02matrix[:,0], prob02matrix[:,1], color = '#417dc0', label = 'car prob. = 0.2') # prob 0.2

plt.ylabel('reward')
plt.xlabel('timestep')
plt.grid(True)
plt.legend()

plt.show()




