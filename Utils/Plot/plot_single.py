from matplotlib import pyplot as plt
import numpy as np
import os

DATA_ROOT_DIR = "../../Output/Single"
DATA_NAMES = ["traveltime.csv", "speed.csv"]
DATA_LABELS = ["traveltime", "speed"]

if __name__ == "__main__":

    for d_idx, data_name in enumerate(DATA_NAMES):
        data_file = os.path.join(DATA_ROOT_DIR, data_name)
        if not os.path.exists(data_file):
            print(f"[Info] (main): The data_file: {data_file} doesn't exist. Skipping...")
            continue

        data = np.loadtxt(data_file)

        x_data = np.arange(1, data.shape[0] + 1)
        for c_idx in range(data.shape[1]):
            y_data = data[:,c_idx].squeeze()
            plt.plot(x_data, y_data, label=DATA_LABELS[d_idx] + str(c_idx))

        plt.legend()
        plt.show()

