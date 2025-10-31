"""vacgrip.py
"""

import struct
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial  # pip install pyserial

import timer

MCU_CLOCK = 53760000
BAUD_RATE = 921600


class VacGripPacketData:
    """VacGripPacketData"""

    DATA_FORMAT: str = (
        "L" + "?" + "hhhhhhh" + "H" + "H" * 32
    )  # tick[1] + flag[1] + imu[7] + loadcell[1] + mic[32]

    header1: bytes
    header2: bytes
    header3: bytes
    reserved: bytes
    data_size: int
    tick: int
    flag: bool
    acc: Tuple[int, int, int]
    gyro: Tuple[int, int, int]
    temperature: int
    loadcell: int
    mic: Tuple[int, ...]
    crc: int

    def __init__(self, packet: Tuple[Any, ...]) -> None:
        self.header1 = packet[0]
        self.header2 = packet[1]
        self.header3 = packet[2]
        self.reserved = packet[3]
        self.data_size = packet[4]
        self.tick = packet[5]
        self.flag = packet[6]
        self.acc = packet[7:10]
        self.gyro = packet[10:13]
        self.temperature = packet[13]
        self.loadcell = packet[14]
        self.mic = packet[15:-1]
        self.crc = packet[-1]

    def __str__(self) -> str:
        text = "{:.4f}: ".format(int(self.tick) / MCU_CLOCK)
        text += str(self.flag) + ", "
        text += str(self.acc) + ", "
        text += str(self.gyro) + ", "
        text += str(self.temperature) + ", "
        text += str(self.loadcell) + ", "
        text += str(self.mic) + ", "
        return text


@dataclass
class VacGripRawData:
    """VacGripRawData"""

    tick: npt.NDArray[np.uint32]
    flag: npt.NDArray[np.bool]
    accelerometer: npt.NDArray[np.int16]
    gyroscope: npt.NDArray[np.int16]
    temperature: npt.NDArray[np.int16]
    loadcell: npt.NDArray[np.uint16]
    microphone: npt.NDArray[np.uint16]


@dataclass
class VacGripData:
    """VacGripData"""

    time: npt.NDArray[np.float64]
    flag: npt.NDArray[np.bool]
    accelerometer: npt.NDArray[np.float64]
    gyroscope: npt.NDArray[np.float64]
    temperature: npt.NDArray[np.float64]
    loadcell: npt.NDArray[np.float64]
    microphone: npt.NDArray[np.float64]


class VacGripPacketHandler:
    """VacGripPacketHandler"""

    # fmt: off
    CRC_TABLE: List[int] = [
        0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011, \
        0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022, \
        0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072, \
        0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041, \
        0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2, \
        0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1, \
        0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1, \
        0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082, \
        0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192, \
        0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1, \
        0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1, \
        0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2, \
        0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151, \
        0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162, \
        0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132, \
        0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101, \
        0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312, \
        0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321, \
        0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371, \
        0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342, \
        0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1, \
        0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2, \
        0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2, \
        0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381, \
        0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291, \
        0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2, \
        0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2, \
        0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1, \
        0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252, \
        0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261, \
        0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231, \
        0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
    ]
    # fmt: on

    def __init__(self) -> None:
        self._data_format = VacGripPacketData.DATA_FORMAT

        self._packet_format = "<cccc"  # header1, header2, header3, reserved
        # little-endian 1bytes 
        self._packet_format += "H"  # data_size
        self._packet_format += self._data_format
        self._packet_format += "H"  # crc

        self._packet_size = struct.calcsize(self._packet_format)

    def _calculate_crc(self, packet: bytes, crc_accum: int = 0):
        crc = crc_accum
        for byte in packet:
            try:
                byte = int(byte)
            except ValueError:
                byte = int(ord(byte))
            idx = (((crc >> 8) & 0xFFFF) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.CRC_TABLE[idx]) & 0xFFFF
        return crc

    def _unpacking_if_valid(self, packet_candidate: bytes) -> Optional[VacGripPacketData]:
        if len(packet_candidate) == self._packet_size:
            if packet_candidate[:4] == b"\xff\xff\xfd\x00":
                packetdata = VacGripPacketData(
                    struct.unpack(self._packet_format, packet_candidate)
                )
                if self._calculate_crc(packet_candidate[:-2]) == packetdata.crc:
                    return packetdata

        return None

    def generate_valid_packet(self, inputbuf: bytearray) -> Iterator[Tuple[bytearray, VacGripPacketData]]:
        """
        입력 버퍼에서 유효한 패킷을 찾아 반환하는 제너레이터.
        del inputbuf[0] 같은 비효율적인 연산을 피하기 위해 find()와 인덱스를 사용합니다.
        """
        start_index = 0
        while True:
            # 패킷 헤더를 찾습니다.
            header_index = inputbuf.find(b"\xff\xff\xfd\x00", start_index)

            # 헤더를 찾지 못하면 루프를 종료합니다.
            if header_index == -1:
                break

            # 패킷 전체가 버퍼에 있는지 확인합니다.
            end_index = header_index + self.packet_size
            if end_index > len(inputbuf):
                break

            # 패킷 후보를 검증합니다.
            packet_candidate = inputbuf[header_index:end_index]
            packetdata = self._unpacking_if_valid(packet_candidate)

            if packetdata:
                yield packet_candidate, packetdata
                start_index = end_index  # 다음 검색 위치를 현재 패킷의 끝으로 이동
            else:
                start_index = header_index + 1 # 헤더는 맞지만 유효하지 않은 패킷. 헤더 다음부터 다시 검색

        # 처리된 부분(start_index 이전)을 버퍼에서 한 번에 제거합니다.
        if start_index > 0:
            del inputbuf[:start_index]

    @property
    def packet_size(self) -> int:
        return self._packet_size


def save_packets(file_path: str, packets: bytearray) -> None:
    with open(file_path, "wb") as file:
        file.write(packets)


def load_packets(file_path: str) -> bytearray:
    with open(file_path, "rb") as file:
        return bytearray(file.read())


def load_rawdata(file_path: str) -> VacGripRawData:
    tick = []
    flag = []
    accelerometer = []
    gyroscope = []
    temperature = []
    loadcell = []
    microphone = []

    for _, packetdata in VacGripPacketHandler().generate_valid_packet(
        load_packets(file_path)
    ):
        tick.append(packetdata.tick)
        flag.append(packetdata.flag)
        accelerometer.append(packetdata.acc)
        gyroscope.append(packetdata.gyro)
        temperature.append(packetdata.temperature)
        loadcell.append(packetdata.loadcell)
        microphone.append(packetdata.mic)

    return VacGripRawData(
        tick=np.array(tick, dtype=np.uint32),
        flag=np.array(flag, dtype=np.bool),
        accelerometer=np.array(accelerometer, dtype=np.int16),
        gyroscope=np.array(gyroscope, dtype=np.int16),
        temperature=np.array(temperature, dtype=np.int16),
        loadcell=np.array(loadcell, dtype=np.uint16),
        microphone=np.array(microphone, dtype=np.uint16),
    )


def load_data(file_path: str) -> VacGripData:
    rawdata = load_rawdata(file_path)

    tick_dif: int = rawdata.tick[1:] - rawdata.tick[:-1] # 50000
    tick_dif[tick_dif < 0] = tick_dif[tick_dif < 0] + 4294967296
    time_dif = tick_dif / 53760000
    
    time = np.cumsum(time_dif)
    time = np.insert(time, 0, 0)

    return VacGripData(
        time=time,
        flag=rawdata.flag,
        accelerometer=rawdata.accelerometer / 8192,
        gyroscope=rawdata.gyroscope / 65.5,
        temperature=rawdata.temperature / 333.87 + 21,
        loadcell=rawdata.loadcell / 4096,
        microphone=rawdata.microphone / 16 / 4096,
    )


class DataWriterState(Enum):
    """State"""

    IDLE = auto()
    COLLECTING = auto()
    FINISH_AND_SAVING = auto()
    SAVED = auto()


class DataWriter:
    """DataWriter"""

    _timer: timer.Timer
    _state: DataWriterState
    _serial: serial.Serial
    _packet_handler: VacGripPacketHandler
    _inputbuf: bytearray
    _packets: bytearray
    _file_path: str

    def __init__(self, port_name: str, baud_rate: int = BAUD_RATE) -> None:
        self._timer = timer.Timer(0.001, self._update)
        self._serial = serial.Serial(port_name, baud_rate, timeout=0)
        self._serial.reset_input_buffer()
        self._packet_handler = VacGripPacketHandler()
        self._clear_buf()

        self._state = DataWriterState.IDLE
        self._timer.start()

    def close(self) -> None:
        if self._timer.is_running():
            self._timer.stop()

    def _clear_buf(self) -> None:
        self._inputbuf = bytearray(0)
        self._packets = bytearray(0)

    def _update(self) -> None:
        self._collect()

        if self._state == DataWriterState.COLLECTING:
            return None

        elif self._state == DataWriterState.FINISH_AND_SAVING:
            self._save()
            self._state = DataWriterState.SAVED

        self._clear_buf()

    def _collect(self) -> None:
        packet_size = self._packet_handler.packet_size

        # add new bytes to input buffer
        while True:
            new_bytes = self._serial.read(packet_size)
            self._inputbuf += new_bytes
            if len(new_bytes) < packet_size:
                break

        # add valid bytes to packets
        for packet, _ in self._packet_handler.generate_valid_packet(self._inputbuf):
            self._packets += packet

    def _save(self) -> None:
        packet_num = len(self._packets) / self._packet_handler.packet_size
        if packet_num == 0:
            print("No packets to save.")

        else:
            save_packets(self._file_path, self._packets)
            print(f'{packet_num} packets are saved on "{self._file_path}".')

    def start(self) -> None:
        print("Start data collection.")
        self._serial.write(b"s")
        self._state = DataWriterState.COLLECTING

    def stop(self) -> None:
        print("Stop data collection and go idle.")
        self._serial.write(b"f")
        self._state = DataWriterState.IDLE

    def finish_and_save(self, file_path: str) -> None:
        print("Finish data collection. Saving...")
        self._serial.write(b"f")
        time.sleep(0.1)

        self._file_path = file_path
        self._state = DataWriterState.FINISH_AND_SAVING

    @property
    def state(self) -> DataWriterState:
        return self._state


class DataMonitor:
    """DataMonitor"""

    _timer: timer.Timer
    _serial: serial.Serial
    _packet_handler: VacGripPacketHandler
    _inputbuf: bytearray
    _packetdata: List[VacGripPacketData]

    def __init__(self, port_name: str, baud_rate: int = BAUD_RATE) -> None:
        self._timer = timer.Timer(1, self._update)
        self._serial = serial.Serial(port_name, baud_rate, timeout=0) # timeout = 0 means non-blocking mode.
        self._serial.reset_input_buffer()
        self._packet_handler = VacGripPacketHandler()
        self._clear_buf()

        self._timer.start()

    def close(self) -> None:
        if self._timer.is_running():
            self._timer.stop()

    def _clear_buf(self) -> None:
        self._inputbuf = bytearray(0)
        self._packetdata = []

    def _update(self) -> None:
        self._collect()

        if len(self._packetdata) != 0:
            for packetdata in self._packetdata:
                print(packetdata)
            self._packetdata.clear()

    def _collect(self) -> None:
        packet_size = self._packet_handler.packet_size

        # add new bytes to input buffer
        while True:
            new_bytes = self._serial.read(packet_size) 
            # Read from serial until packet_size and because the serial is non-blocking mode 
            # The serial buffer can be less then packet_size.
            self._inputbuf += new_bytes
            if len(new_bytes) < packet_size:
                # This means that serial buffer is empty.
                break

        # add valid bytes to packets
        for _, packetdata in self._packet_handler.generate_valid_packet(self._inputbuf):
            self._packetdata.append(packetdata)
