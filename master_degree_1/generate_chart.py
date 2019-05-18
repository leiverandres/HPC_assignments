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

if __name__ == "__main__":
    seq_times_collector = parse_times_file(seq_time_filename)
    th_times_collector = parse_times_file(threads_time_filename)
    seq_sizes, seq_times = get_mean_times(seq_times_collector)
    th_sizes, th_times = get_mean_times(th_times_collector)
    plt.plot(seq_sizes, seq_times, color="#ffa600")
    plt.plot(th_sizes, th_times)
    plt.ylabel("Mean time (sec)")
    plt.xlabel("Size of square matrix")
    plt.savefig("times.png")
    plt.show()
