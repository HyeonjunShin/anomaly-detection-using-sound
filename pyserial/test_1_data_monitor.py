"""test_1_data_monitor.py
"""

import time
import serial.tools.list_ports
import serial # serial.SerialException을 사용하기 위해 import
import vacgrip

# --- 설정 변경: Linux 환경의 직렬 포트 경로를 직접 지정 ---
SERIAL_PORT = "/dev/ttyUSB0"
# ----------------

if __name__ == "__main__":
    print("사용 가능한 직렬 포트 목록:")
    # serial.tools.list_ports.comports()를 사용하여 사용 가능한 포트 목록을 보여줍니다.
    infos = serial.tools.list_ports.comports()
    for info in infos:
        print(info.name + ": " + info.description)

    print(f"\n데이터 모니터를 {SERIAL_PORT}로 연결을 시도합니다...")

    # DataMonitor 객체 생성 및 연결 시도
    try:
        # COM 대신 /dev/ttyUSB0 경로 사용
        monitor = vacgrip.DataMonitor(
            port_name=SERIAL_PORT
        )
    except serial.SerialException as e:
        # vacgrip.DataMonitor 내부에서 발생한 시리얼 연결 오류 처리
        print(f"❌ 오류: {SERIAL_PORT} 연결에 실패했습니다.")
        print("  - 장치가 연결되어 있는지 확인하세요.")
        print("  - 현재 사용자가 'dialout' 그룹에 속해 있는지 확인하세요. (필요 시 재로그인)")
        # 상세 오류 메시지 출력
        # print(f"상세 오류: {e}") 
        exit(1)


    print(f"✅ {SERIAL_PORT} 연결 성공. 데이터 수신 대기 중. (Ctrl+C로 종료)")

    while True:
        try:
            # DataMonitor 내부의 timer가 데이터를 수신하고 출력합니다.
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n키보드 인터럽트 감지.")
            break

    print("프로그램 종료 중...")
    monitor.close()
    print("연결이 닫혔습니다.")