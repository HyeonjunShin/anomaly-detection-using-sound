import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json


DATA_PATH = "./data/csv_origin/"
JSON_PATH = "./data/json/"

def process_data(timings, success_data, fail_data):

    succs = []
    files = []
    for timing, succ, fail in zip(timings, success_data, fail_data):
        start_time, end_time = timing
        succs += [succ[start_time:end_time]]
        files += [fail[start_time:end_time]]


    # Find the minimum length
    min_len = min(len(d) for d in succs + files)
    
    # Trim all data to the minimum length
    succs = [d[:min_len] for d in succs]
    files = [d[:min_len] for d in files]
    
    mean_success = np.mean(succs, axis=0)
    mean_fail = np.mean(files, axis=0)
    difference = mean_success - mean_fail

    # Standardize the difference
    diff_mean = np.mean(difference)
    diff_std = np.std(difference)
    difference = (difference - diff_mean) / (diff_std + 1e-8)

    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(mean_success.T, aspect='auto', origin='lower', norm=colors.LogNorm())
    plt.title("Average Spectrogram (Success)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Amplitude")

    plt.subplot(1, 3, 2)
    plt.imshow(mean_fail.T, aspect='auto', origin='lower', norm=colors.LogNorm())
    plt.title("Average Spectrogram (Failure)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Amplitude")

    v_max = np.percentile(difference, 98)
    v_min = np.percentile(difference, 2)

    plt.subplot(1, 3, 3)
    # plt.imshow(difference.T, aspect='auto', origin='lower', cmap='coolwarm', norm=colors.SymLogNorm(linthresh=0.01, vmin=v_min, vmax=v_max))
    plt.imshow(difference.T, aspect='auto', origin='lower', cmap='coolwarm')
    plt.title("Difference (Success - Failure)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Amplitude Difference")

    plt.tight_layout()
    plt.show()

def read_json(BASE_PATH):
    paths = [p for p in os.listdir(BASE_PATH) if p.endswith("json")]
    
    ret = []
    for path in paths:
        with open(BASE_PATH + path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            start_time = data["데이터 시점"]["Scution on 시점"]
            end_time = data["데이터 시점"]["최대 하강 시점"]

            ret += [[start_time, end_time]]
    return ret


def read_data(BASE_PATH):
    paths = os.listdir(BASE_PATH)
    
    success_data = []
    fail_data = []

    for path in paths:
        if ".csv" not in path:
            continue

        splited_name = path.split(" ")
        
        try:
            csv_file = pd.read_csv(BASE_PATH + path, header=None)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        mic_data = csv_file.iloc[:, 7:39].to_numpy() # collect only mic data channels

        if "성공" in splited_name[1]:
            success_data.append(mic_data)
        elif "실패" in splited_name[1]:
            fail_data.append(mic_data)

    if not success_data or not fail_data:
        print("Not enough data to compare.")
        return None, None
    
    return success_data, fail_data


def main():
    timings = read_json(JSON_PATH)
    success_data, fail_data = read_data(DATA_PATH)
    process_data(timings, success_data, fail_data)

if __name__ == "__main__":
    main()