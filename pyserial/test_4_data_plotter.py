"""test_4_data_plotter.py
"""

import os

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

# --- 설정 변경 ---
# 1. 원본 데이터 파일이 저장된 절대 경로
SAVE_DIR = "/home/jeong/test/pyserial/data"
# 2. 플로팅할 .npz 파일의 이름 (확장자 제외)
FILE_NAME = "20251030_083311"
# -----------------


if __name__ == "__main__":
    
    # 3. .npz 파일의 전체 경로 생성
    data_file_path = os.path.join(SAVE_DIR, FILE_NAME + ".npz")
    
    # 파일 존재 여부 확인
    if not os.path.exists(data_file_path):
        print(f"❌ 오류: .npz 파일이 경로에 존재하지 않습니다: {data_file_path}")
        exit(1)

    print(f"데이터 로드 중: {data_file_path}")
    
    # np.load를 사용하여 데이터 로드
    npzfile = np.load(data_file_path)

    # NumPy LoadFile.f 객체를 data 변수에 할당 (이전 코드 구조 유지)
    data = npzfile.f
    print(f"data length: {len(data.time)}")

    # plot data
    plt.figure(1, figsize=(10, 8)) # Figure 크기 조정 (가독성 향상)

    plt.subplot(4, 1, 1)
    # plt.plot(data.time) -> x축을 인덱스 대신 시간으로 사용하도록 수정
    plt.plot(data.time, data.time) 
    plt.xlim(min(data.time), max(data.time))
    plt.title("Time (s)")
    plt.grid(True)
    plt.ylabel("Time (s)")

    plt.subplot(4, 1, 2)
    plt.plot(data.time[1:], data.time[1:] - data.time[:-1])
    plt.xlim(min(data.time), max(data.time))
    plt.title("Time Difference (s)")
    plt.grid(True)
    plt.ylabel("Delta Time (s)")

    plt.subplot(4, 1, 3)
    plt.plot(data.time, data.temperature)
    plt.xlim(min(data.time), max(data.time))
    plt.title("Temperature (C)")
    plt.grid(True)
    plt.ylabel("Temp (C)")

    plt.subplot(4, 1, 4)
    plt.plot(data.time, data.flag)
    plt.xlim(min(data.time), max(data.time))
    plt.title("Flag")
    plt.grid(True)
    plt.ylabel("Flag Value")
    plt.xlabel("Time (s)")

    plt.tight_layout() # 서브플롯 간 간격 자동 조정


    plt.figure(2, figsize=(10, 10)) # Figure 크기 조정

    plt.subplot(4, 1, 1)
    # data.accelerometer가 (N, 3) 형태라고 가정하고 세 축을 모두 플로팅
    plt.plot(data.time, data.accelerometer[:, 0], label="acc_x")
    plt.plot(data.time, data.accelerometer[:, 1], label="acc_y")
    plt.plot(data.time, data.accelerometer[:, 2], label="acc_z")
    plt.legend()
    plt.xlim(min(data.time), max(data.time))
    plt.title("Accelerometer (g)")
    plt.grid(True)
    plt.ylabel("Acceleration (g)")

    plt.subplot(4, 1, 2)
    # data.gyroscope가 (N, 3) 형태라고 가정하고 세 축을 모두 플로팅
    plt.plot(data.time, data.gyroscope[:, 0], label="gyro_x")
    plt.plot(data.time, data.gyroscope[:, 1], label="gyro_y")
    plt.plot(data.time, data.gyroscope[:, 2], label="gyro_z")
    plt.legend()
    plt.xlim(min(data.time), max(data.time))
    plt.title("Gyroscope (deg/s)")
    plt.grid(True)
    plt.ylabel("Angular Rate (deg/s)")

    plt.subplot(4, 1, 3)
    plt.plot(data.time, data.loadcell, label="loadcell")
    plt.xlim(min(data.time), max(data.time))
    plt.title("Loadcell")
    plt.grid(True)
    plt.ylabel("Loadcell Value")

    plt.subplot(4, 1, 4)
    # data.microphone이 (N, 32) 형태라고 가정
    mic_plot = plt.imshow(
        data.microphone.T, # 전치하여 시간축이 x축이 되도록 함
        aspect="auto",
        cmap="viridis",
        origin="lower",
        norm=colors.LogNorm(vmin=1, vmax=data.microphone.max()), # LogNorm의 vmin 설정
        extent=[min(data.time), max(data.time), 0, data.microphone.shape[1]] # x축을 time으로 설정
    )
    plt.colorbar(mic_plot, ax=plt.gca(), label='Microphone Amplitude (Log Scale)')
    plt.title("Microphone Data (Heatmap)")
    plt.ylabel("Mic Channel Index")
    plt.xlabel("Time (s)")

    plt.tight_layout() # 서브플롯 간 간격 자동 조정

    plt.show()