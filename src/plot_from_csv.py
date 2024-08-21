import csv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

CSV_FILE_NAME_005 = 'csv_data_/test_pr_0.05.csv'
CSV_FILE_NAME_01 = 'csv_data_/test_pr_0.1.csv'
CSV_FILE_NAME_02 = 'csv_data_/test_pr_0.2.csv'
CSV_FILE_NAME_04 = 'csv_data_/test_pr_0.4.csv'

csvs = [CSV_FILE_NAME_005, CSV_FILE_NAME_02, CSV_FILE_NAME_04]
#csvs = [CSV_FILE_NAME_01, CSV_FILE_NAME_02]
csvLengths = []

for currCsv in csvs:
    csvLengths.append(int(len(currCsv)))

minCsvLength = min(csvLengths)

# Read the CSV file into a DataFrame
for currCsv in csvs:
    dataFrameCsv = pd.read_csv(currCsv)
    dataFrameCsv = dataFrameCsv[:-minCsvLength]
    # Save the modified DataFrame back to the CSV file
    dataFrameCsv.to_csv(currCsv, index=False)


prob005matrix = np.loadtxt(CSV_FILE_NAME_005, delimiter=",", dtype = float,skiprows=1)
prob02matrix = np.loadtxt(CSV_FILE_NAME_02, delimiter=",", dtype = float,skiprows=1)
prob04matrix = np.loadtxt(CSV_FILE_NAME_04, delimiter=",", dtype = float,skiprows=1)




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
# 
#    order:  Timestep,Reward,Num Vehicles,Speed,Waiting time
#               0       1     2             3      4        


#fig, ax1 = plt.subplots()

#mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])


# reward

plt.plot(prob005matrix[:,0], prob005matrix[:,1], color = 'indianred', linewidth = '1', label = 'car prob. = 0.05') # prob 0.05
#plt.plot(prob02matrix[:,0], prob02matrix[:,1], color = 'mediumseagreen',  linewidth = '1', label = 'car prob. = 0.2') # prob 0.2
#plt.plot(prob02matrix[:,0], prob04matrix[:,1], color = 'royalblue',  linewidth = '1', label = 'car prob. = 0.4') # prob 0.4

plt.ylabel('reward')
plt.xlabel('timestep')
plt.grid(True)
plt.legend()

plt.show()

# num veh

plt.plot(prob005matrix[:,0], prob005matrix[:,2], color = 'indianred', linewidth = '1', label = 'car prob. = 0.05') # prob 0.05
#plt.plot(prob02matrix[:,0], prob02matrix[:,2], color = 'mediumseagreen',  linewidth = '1', label = 'car prob. = 0.2') # prob 0.2
#plt.plot(prob02matrix[:,0], prob04matrix[:,2], color = 'royalblue',  linewidth = '1', label = 'car prob. = 0.4') # prob 0.4

plt.ylabel('# of vehicles')
plt.xlabel('timestep')
plt.grid(True)
plt.legend()

plt.show()



# sped

plt.plot(prob005matrix[:,0], prob005matrix[:,3], color = 'indianred', linewidth = '1', label = 'car prob. = 0.05') # prob 0.05
#plt.plot(prob02matrix[:,0], prob02matrix[:,3], color = 'mediumseagreen',  linewidth = '1', label = 'car prob. = 0.2') # prob 0.2
#plt.plot(prob02matrix[:,0], prob04matrix[:,3], color = 'royalblue',  linewidth = '1', label = 'car prob. = 0.4') # prob 0.4

plt.ylabel('average speed')
plt.xlabel('timestep')
plt.grid(True)
plt.legend()

plt.show()




# waittiem

plt.plot(prob005matrix[:,0], prob005matrix[:,4], color = 'indianred', linewidth = '1', label = 'car prob. = 0.05') # prob 0.05
#plt.plot(prob02matrix[:,0], prob02matrix[:,4], color = 'mediumseagreen',  linewidth = '1', label = 'car prob. = 0.2') # prob 0.2
#plt.plot(prob02matrix[:,0], prob04matrix[:,4], color = 'royalblue',  linewidth = '1', label = 'car prob. = 0.4') # prob 0.4

plt.ylabel('waittime')
plt.xlabel('timestep')
plt.grid(True)
plt.legend()

plt.show()



