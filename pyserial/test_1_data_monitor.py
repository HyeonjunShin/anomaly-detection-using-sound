"""test_1_data_monitor.py
"""

import time

import serial.tools.list_ports
import vacgrip

if __name__ == "__main__":
    print("List of COM ports")
    infos = serial.tools.list_ports.comports()
    for info in infos:
        print(info.name + ": " + info.description)

    monitor = vacgrip.DataMonitor(
        # port_name="COM" + input("Type the number of COM port:")
        port_name="/dev/" + infos[int(input("Type the number of COM port:"))].name
    )

    while True:
        try:
            time.sleep(0.1)

        except KeyboardInterrupt:
            break

    monitor.close()
