"""main.py - 's' 입력 시 모니터링/수집 동시 시작 프로그램 (마이크로폰 전용 플롯)"""

import datetime
import os
import time
import sys
import select 

import serial
import serial.tools.list_ports
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
import vacgrip 

# **********************************************
# --- 1. 설정 변경 ---
# **********************************************
SERIAL_PORT = "/dev/ttyUSB0"
ROOT_SAVE_DIR = "/home/jeong/test/pyserial/data"
# 플롯 저장 시 사용할 이미지 파일명 (하나로 통일)
PLOT_FILE_NAME = "data_mic_plot.png"
# ----------------------------------------------

def setup_serial_connection(DataHandlerClass):
    try:
        handler = DataHandlerClass(port_name=SERIAL_PORT) 
        return handler
    except serial.SerialException:
        exit(1)

def get_non_blocking_input():
    """Linux 환경에서 키 입력을 비동기로 읽어옵니다."""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.readline().strip()
    return None

def convert_data(dat_file_path):
    """test_3_data_converter 기능을 실행합니다."""
    print("\n[변환 단계] 데이터 변환을 시작합니다...")
    base_dir = os.path.dirname(dat_file_path)
    file_name_base = os.path.splitext(os.path.basename(dat_file_path))[0]
    npz_file_path = os.path.join(base_dir, file_name_base + ".npz")
    if not os.path.exists(dat_file_path):
        print(f"❌ 오류: 원본 파일이 경로에 존재하지 않습니다: {dat_file_path}")
        return None
    try:
        data = vacgrip.load_data(dat_file_path)
        # 마이크로폰 데이터만 저장하도록 수정
        np.savez_compressed(
            npz_file_path,
            time=data.time, 
            flag=data.flag, 
            microphone=data.microphone,
        )
        print(f"✅ 변환 및 저장 완료: {npz_file_path}")
        return npz_file_path
    except Exception as e:
        print(f"❌ 데이터 로드 또는 저장 중 오류 발생:"); print(f"상세 오류: {e}"); return None

# ----------------------------------------------
# --- plot_data 함수: 시간 및 마이크로폰 데이터만 남김 ---
# ----------------------------------------------
def plot_data(npz_file_path, current_save_dir):
    """시간 및 마이크로폰 데이터만 플롯하고 이미지를 저장합니다."""
    print("\n[플롯 단계] 데이터를 시각화하고 이미지로 저장합니다...")
    
    try:
        npzfile = np.load(npz_file_path)
        data = npzfile.f
        
        if not hasattr(data, 'microphone') or len(data.time) == 0:
             print("❌ 오류: NPZ 파일에 마이크로폰 데이터가 없거나 비어 있습니다.")
             return

        print(f"데이터 길이: {len(data.time)} 샘플")
                
        # Subplot 2: Microphone Data (Heatmap)
        plt.plot(1);
        
        mic_data = data.microphone
            
        mic_plot = plt.imshow(
            mic_data.T, aspect="auto", cmap="viridis", origin="lower",
            norm=colors.LogNorm(),
            extent=[min(data.time), max(data.time), 0, mic_data.shape[1]]
        )
        plt.colorbar(mic_plot, ax=plt.gca(), label=f'Microphone Amplitude (Log Scale, Range:)'); 
        plt.title("Microphone Data (Heatmap)"); 
        plt.ylabel("Mic Channel Index"); 
        plt.xlabel("Time (s)"); 
        
        plt.tight_layout()
        
        # 저장
        image_file_path = os.path.join(current_save_dir, PLOT_FILE_NAME)
        plt.savefig(image_file_path)
        print(f"✅ 플롯 이미지 저장 완료: {image_file_path}")
        
        plt.show()

    except Exception as e:
        print(f"❌ 플롯 생성/저장 중 오류 발생:"); print(f"상세 오류: {e}")

# ----------------------------------------------
# --- run_main_loop 및 __main__ 블록 (이전과 동일) ---
# ----------------------------------------------

def run_main_loop(monitor, current_save_dir):
    """키 입력 시 모니터링/수집 동시 시작을 처리하는 메인 루프."""
    
    # 1. 저장 디렉토리 생성
    try:
        os.makedirs(current_save_dir, exist_ok=True)
        print(f"⭐ 데이터 저장 경로: {current_save_dir}")
    except OSError as e:
        print(f"❌ 오류: 저장 디렉토리 생성 실패. 경로 권한을 확인하세요: {current_save_dir}"); print(f"상세 오류: {e}")
        return None, False 

    print('----------------------------------------------------')
    print("⭐ [모니터링 대기] 's'를 눌러 데이터 수집 및 모니터링을 시작하세요. ⭐")
    print('  "s": 수집 시작 (모니터링 동시 시작)')
    print('  "f": 수집 중단, 파일 저장 및 후처리 (종료)')
    print('  "q": 프로그램 종료 (저장 안 함)')
    print('----------------------------------------------------')

    dat_file_path = None
    
    while True:
        try:
            # 1. 키 입력 감지 (비동기)
            text = get_non_blocking_input()
            
            if text:
                print(f"입력: {text}")
                
                if text == "s":
                    if monitor.state == vacgrip.DataWriterState.COLLECTING:
                        print("이미 수집 중입니다.")
                        continue
                    monitor.start() 
                    print("--> 데이터 수집/모니터링 동시 시작.")

                elif text == "f":
                    if monitor.state != vacgrip.DataWriterState.COLLECTING:
                        print("수집 중이 아닙니다. 's'를 입력하여 먼저 수집을 시작하세요.")
                        continue
                        
                    file_name_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dat_file_path = os.path.join(current_save_dir, file_name_base + ".dat")
                    
                    print(f"--> 데이터 수집 중단 및 '{dat_file_path}'에 저장 요청.")
                    monitor.finish_and_save(dat_file_path)
                    
                    while monitor.state != vacgrip.DataWriterState.SAVED:
                        time.sleep(0.1)
                        
                    print("--> 파일 저장 완료. 후처리 진행...")
                    return dat_file_path, True # 저장 완료 및 성공

                elif text in ["q", "quit"]:
                    return None, False # 프로그램 종료

                else:
                    print('유효하지 않은 명령어입니다.')

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n키보드 인터럽트 감지. 프로그램 종료."); return None, False

# ----------------------------------------------

if __name__ == "__main__":
    
    print("사용 가능한 직렬 포트 목록:")
    infos = serial.tools.list_ports.comports()
    for info in infos:
        print(info.name + ": " + info.description)

    monitor = setup_serial_connection(vacgrip.DataMonitor)
    
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # current_save_dir = os.path.join(ROOT_SAVE_DIR, timestamp)

    # dat_file_path, success = None, False
    # if monitor is not None:
    #     dat_file_path, success = run_main_loop(monitor, current_save_dir)
    #     monitor.close()
        
    # if dat_file_path and success:
    #     npz_file_path = convert_data(dat_file_path)
    #     if npz_file_path:
    #         plot_data(npz_file_path, current_save_dir)
            
    # print("\n\n--- 프로그램 전체 종료 ---")
    # print("모든 연결이 닫혔습니다.")