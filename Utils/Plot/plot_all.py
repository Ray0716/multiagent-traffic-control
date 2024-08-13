from matplotlib import pyplot as plt
import os
import numpy as np

DATA_ROOT_PATH = os.getcwd()
DATA_SIZE = 100
DATA_NAMES = ["traveltime.csv", "score.csv"]
DATA_LABELS = ["Mean travel time", "Mean rewards"]
CONGESTION_NAMES = ['0p05', '0p1', '0p2', '0p4']
CONGESTION_LABELS = ['q = 0.05', 'q = 0.1', 'q = 0.2', 'q = 0.4']

def plotData(data_root_path, data_name, data_size, plot_info):
    mean_arr = []
    std_arr = []
    for idx in range(1, data_size+1):
        data_dir = os.path.join(data_root_path, str(idx))
        data_file = os.path.join(data_dir, data_name)
        if not os.path.exists(data_file):
            continue

        data = np.loadtxt(data_file)
        data_mean = np.mean(data)
        mean_arr.append(data_mean)
        data_std = np.std(data)
        std_arr.append(data_std)

    mean_arr = np.array(mean_arr)
    std_arr = np.array(std_arr)

    # Plot mean
    data_size = min(data_size, len(mean_arr))
    x_data = [item for item in range(1, data_size + 1)]
    plt.plot(x_data, mean_arr, '-', label=plot_info['label_mean'])

    # Plot mean constant
    if plot_info.get('color_mean_constant') is not None and plot_info.get('label_mean_constant') is not None:
        mean_total = np.mean(mean_arr)
        plt.plot(x_data, [mean_total]*data_size, '-', c=plot_info['color_mean_constant'], label=plot_info['label_mean_constant'].format(mean_total=mean_total))

    # Plot std
    sigma_multiplier = plot_info.get('std_multiplier', 1)
    upper_bound = mean_arr + sigma_multiplier * std_arr  # mean plus two standard deviations
    lower_bound = mean_arr - sigma_multiplier * std_arr  # mean minus two standard deviations
    # Add shaded region to the plot
    if plot_info.get('label_std') is not None:
        plt.fill_between(x_data, lower_bound, upper_bound, color=plot_info['color_std'], alpha=0.5, label=plot_info['label_std'])
    else:
        plt.fill_between(x_data, lower_bound, upper_bound, color=plot_info['color_std'], alpha=0.5)

if __name__ == "__main__":
    plot_info = {
        'label_mean' : 'Low',
        'color_std' : 'grey', 'label_std' : '± $\sigma$',
        # 'color_mean_constant' : 'green', 'label_mean_constant' : 'mean: {mean_total:.2f}',
    }
    for d_idx, data_name in enumerate(DATA_NAMES):
        for c_idx, congestion_name in enumerate(CONGESTION_NAMES):
            data_root_path = os.path.join(DATA_ROOT_PATH, congestion_name, 'Results')
            if not os.path.isdir(data_root_path):
                print(f"[Info] (main): data_root_path: {data_root_path} doesn't exist. Skipping...")
                print(f"[Info] (main): data_name: {data_name}, congestion_name: {congestion_name}")
                print()
                continue
            plot_info['label_mean'] = CONGESTION_LABELS[c_idx]
            plotData(data_root_path, data_name, DATA_SIZE, plot_info)
            print()

        plt.xlabel('Time steps')
        plt.ylabel(DATA_LABELS[d_idx])
        plt.grid()
        plt.legend()

        plt.show()
