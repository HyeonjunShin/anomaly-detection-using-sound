"""test_2_data_writer.py
"""

import datetime
import os

import serial.tools.list_ports
import vacgrip

if __name__ == "__main__":
    print("List of COM ports")
    infos = serial.tools.list_ports.comports()
    for info in infos:
        print(info.name + ": " + info.description)

    data_writer = vacgrip.DataWriter(
        port_name="COM" + input("Type the number of COM port:")
    )

    print('Enter "s" to start, "x" to stop_and_clear, "f" to finish_and_save.')

    while True:
        try:
            text = input()
            if text == "s":
                data_writer.start()

            elif text == "x":
                data_writer.stop()

            elif text == "f":
                file_path: str = os.path.join(
                    os.path.dirname(__file__),
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                data_writer.finish_and_save(file_path)

        except KeyboardInterrupt:
            break

    data_writer.close()
