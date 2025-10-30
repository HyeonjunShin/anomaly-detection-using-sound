"""test_3_data_converter.py
"""

import os

import numpy as np
import vacgrip

FILE_NAME = "20230810_235736_120sec_sample"


if __name__ == "__main__":
    packet_file_path = os.path.join(os.path.dirname(__file__), FILE_NAME)
    data = vacgrip.load_data(packet_file_path)
    np.savez_compressed(
        packet_file_path + ".npz",
        time=data.time,
        flag=data.flag,
        accelerometer=data.accelerometer,
        gyroscope=data.gyroscope,
        temperature=data.temperature,
        loadcell=data.loadcell,
        microphone=data.microphone,
    )
