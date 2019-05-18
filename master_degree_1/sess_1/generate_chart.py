import numpy as np
import matplotlib.pyplot as plt


def parse_times_file(filename):
    times_collector = {}
    with open(filename, "r") as times_file:
        data = times_file.readlines()
        for line in data[1:]:
            line_splited = line.split(",")
            size = int(line_splited[0].strip())
            time = float(line_splited[1].strip())
            if not size in times_collector:
                times_collector[size] = np.array([])
            times_collector[size] = np.append(times_collector[size], time)
    return times_collector


def get_mean_times(times_by_size):
    sizes = []
    mean_times = []
    for size, times in times_by_size.items():
        sizes.append(size)
        mean_times.append(times.mean())
    return sizes, mean_times


seq_time_filename = "./times.txt"
threads_time_filename = "./th_times.txt"
process_time_filename = "./proc_times.txt"

if __name__ == "__main__":
    files = [
        # {"file": seq_time_filename, "chartName": "sequential"},
        {"file": threads_time_filename, "chartName": "threads"},
        # {"file": process_time_filename, "chartName": "processes"},
    ]
    for fileinfo in files:
        print(f"Reading {fileinfo['file']}")
        times_collector = parse_times_file(fileinfo["file"])
        sizes, times = get_mean_times(times_collector)
        print(list(zip(sizes, times)))
        plt.plot(sizes, times, color="orange")
    plt.legend([fileinfo["chartName"] for fileinfo in files])
    plt.ylabel("Mean time (sec)")
    plt.xlabel("Size of square matrix")
    plt.savefig("times_threads.png")
    plt.show()
