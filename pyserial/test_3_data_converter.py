"""test_3_data_converter.py
"""

import os
import numpy as np
import vacgrip

# --- 설정 변경 ---
# 1. 원본 데이터 파일이 저장된 절대 경로
SAVE_DIR = "/home/jeong/test/pyserial/data"
# 2. 변환할 원본 데이터 파일의 이름 (확장자 제외)
FILE_NAME = "20251030_083311" 
# -----------------

if __name__ == "__main__":
    
    # 3. 원본 패킷 파일의 전체 경로 (확장자 .dat 추가)
    packet_file_path_dat = os.path.join(SAVE_DIR, FILE_NAME + ".dat")
    
    # 변환된 npz 파일을 저장할 경로 (확장자 .npz 추가)
    npz_file_path = os.path.join(SAVE_DIR, FILE_NAME + ".npz")
    
    print(f"변환을 시작합니다: {packet_file_path_dat}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(packet_file_path_dat):
        print(f"❌ 오류: 원본 파일이 경로에 존재하지 않습니다: {packet_file_path_dat}")
        exit(1)

    try:
        # vacgrip.load_data 함수를 사용하여 .dat 파일에서 데이터를 로드
        data = vacgrip.load_data(packet_file_path_dat)
        
        # NumPy 압축 파일 (.npz)로 저장
        np.savez_compressed(
            npz_file_path,
            time=data.time,
            flag=data.flag,
            accelerometer=data.accelerometer,
            gyroscope=data.gyroscope,
            temperature=data.temperature,
            loadcell=data.loadcell,
            microphone=data.microphone,
        )
        
        print(f"✅ 변환 및 저장 완료: {npz_file_path}")

    except Exception as e:
        print(f"❌ 데이터 로드 또는 저장 중 오류 발생:")
        print(f"상세 오류: {e}")