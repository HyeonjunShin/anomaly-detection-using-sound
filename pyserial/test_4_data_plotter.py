"""test_4_data_plotter.py
"""

import os

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

FILE_NAME = "20230810_235736_120sec_sample"


if __name__ == "__main__":
    data_file_path = os.path.join(os.path.dirname(__file__), FILE_NAME) + ".npz"
    npzfile = np.load(data_file_path)

    data = npzfile.f
    print(f"data length: {len(data.time)}")

    # plot data
    plt.figure(1)

    plt.subplot(4, 1, 1)
    plt.plot(data.time)
    plt.xlim(0, len(data.time))
    plt.title("time (s)")

    plt.subplot(4, 1, 2)
    plt.plot(data.time[1:], data.time[1:] - data.time[:-1])
    plt.xlim(min(data.time), max(data.time))
    plt.title("time dif (s)")

    plt.subplot(4, 1, 3)
    plt.plot(data.time, data.temperature)
    plt.xlim(min(data.time), max(data.time))
    plt.title("temperature (C)")

    plt.subplot(4, 1, 4)
    plt.plot(data.time, data.flag)
    plt.xlim(min(data.time), max(data.time))
    plt.title("flag")

    plt.figure(2)

    plt.subplot(4, 1, 1)
    plt.plot(data.time, data.accelerometer, label="acc_x")
    plt.legend()
    plt.xlim(min(data.time), max(data.time))
    plt.title("accelerometer (g)")

    plt.subplot(4, 1, 2)
    plt.plot(data.time, data.gyroscope, label="gyro_x")
    plt.legend()
    plt.xlim(min(data.time), max(data.time))
    plt.title("gyroscope (deg/s)")

    plt.subplot(4, 1, 3)
    plt.plot(data.time, data.loadcell, label="loadcell")
    plt.xlim(min(data.time), max(data.time))
    plt.title("loadcell")

    plt.subplot(4, 1, 4)
    plt.imshow(
        data.microphone.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        norm=colors.LogNorm(),
    )
    plt.xlim(0, len(data.microphone))
    # plt.colorbar()
    plt.title("mic")

    plt.show()
