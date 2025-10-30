"""test_2_data_writer.py
"""

import datetime
import os
import serial # serial.SerialException 처리를 위해 import
import serial.tools.list_ports
import vacgrip

# --- 설정 변경 ---
# 1. Linux 환경의 직렬 포트 경로를 직접 지정
SERIAL_PORT = "/dev/ttyUSB0"
# 2. 지정된 절대 저장 경로로 수정
SAVE_DIR = "/home/jeong/test/pyserial/data"
# -----------------

if __name__ == "__main__":
    print("사용 가능한 직렬 포트 목록:")
    infos = serial.tools.list_ports.comports()
    for info in infos:
        print(info.name + ": " + info.description)

    print(f"\n데이터 라이터를 {SERIAL_PORT}로 연결을 시도합니다...")

    # DataWriter 객체 생성 및 연결 시도
    try:
        # /dev/ttyUSB0 경로 사용
        data_writer = vacgrip.DataWriter(
            port_name=SERIAL_PORT
        )
    except serial.SerialException:
        # 시리얼 연결 오류 처리
        print(f"❌ 오류: {SERIAL_PORT} 연결에 실패했습니다.")
        print("  - 장치가 연결되어 있는지, 권한이 있는지 확인하세요.")
        exit(1)

    print(f"✅ {SERIAL_PORT} 연결 성공. 준비 완료.")
    print('----------------------------------------------------')
    print('Enter "s" to start, "x" to stop_and_clear, "f" to finish_and_save.')
    print(f"⭐ 데이터는 다음 경로에 저장됩니다: {SAVE_DIR}")

    # 저장 디렉토리가 없으면 생성 (경로 권한 문제 발생 가능성 주의)
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
    except OSError as e:
        print(f"❌ 오류: 저장 디렉토리 생성 실패. 경로 권한을 확인하세요: {SAVE_DIR}")
        print(f"상세 오류: {e}")
        data_writer.close()
        exit(1)

    while True:
        try:
            text = input("> ") # 사용자 입력 프롬프트

            if text == "s":
                if data_writer.state == vacgrip.DataWriterState.COLLECTING:
                    print("이미 수집 중입니다.")
                    continue
                data_writer.start()

            elif text == "x":
                if data_writer.state != vacgrip.DataWriterState.COLLECTING:
                    print("수집 중이 아닙니다.")
                    continue
                data_writer.stop()
                print("수집 정지 및 버퍼 초기화 완료.")

            elif text == "f":
                if data_writer.state != vacgrip.DataWriterState.COLLECTING:
                    print("수집 중이 아닙니다. 's'를 입력하여 먼저 수집을 시작하세요.")
                    continue
                    
                # 파일 경로 생성: SAVE_DIR/YYYYMMDD_HHMMSS.dat
                file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.dat")
                file_path = os.path.join(SAVE_DIR, file_name)
                
                print(f"데이터 수집을 중단하고 '{file_path}'에 저장합니다.")
                data_writer.finish_and_save(file_path)

            elif text in ["q", "quit"]:
                break
                
            else:
                print('유효하지 않은 명령어입니다. "s", "x", "f", "q" 중 하나를 입력하세요.')

        except KeyboardInterrupt:
            break

    print("\n프로그램 종료 중...")
    data_writer.close()
    print("연결이 닫혔습니다.")