from matplotlib import pyplot as plt
import os
import numpy as np

DATA_ROOT_PATH = os.getcwd()
DATA_SIZE = 100
DATA_NAME = "traveltime.csv"

mean_arr = []
std_arr = []
for idx in range(1, DATA_SIZE+1):
    data_dir = os.path.join(DATA_ROOT_PATH, str(idx))
    data_file = os.path.join(data_dir, DATA_NAME)

    data = np.loadtxt(data_file)
    data_mean = np.mean(data)
    mean_arr.append(data_mean)
    data_std = np.std(data)
    std_arr.append(data_std)

mean_arr = np.array(mean_arr)
std_arr = np.array(std_arr)

# Plot mean
x_data = [item for item in range(1, DATA_SIZE + 1)]
plt.plot(x_data, mean_arr, '-', label='Low')

# Plot mean constant
mean_total = np.mean(mean_arr)
plt.plot(x_data, [mean_total]*DATA_SIZE, '-', c='green', label=f'mean: {mean_total:.2f}')

# Plot std
upper_bound = mean_arr + 2 * std_arr  # mean plus two standard deviations
lower_bound = mean_arr - 2 * std_arr  # mean minus two standard deviations
# Add shaded region to the plot
# plt.fill_between(x_data, lower_bound, upper_bound, color='gray', alpha=0.5)
plt.fill_between(x_data, mean_arr + std_arr, mean_arr - std_arr, color='gray', alpha=0.5, label='± $\sigma$')

# Add labels and title, if necessary
plt.xlabel('Time steps')
plt.ylabel('Mean travel time')
plt.grid()
plt.legend()

plt.show()
