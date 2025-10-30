"""vacgrip.py
데이터 패킷 처리, 로드 및 시리얼 통신 클래스 정의 (마이크로폰 전용으로 수정됨)
"""

import struct
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial  # pip install pyserial

import timer # timer 모듈이 현재 환경에 정의되어 있다고 가정

MCU_CLOCK = 53760000
BAUD_RATE = 921600


class VacGripPacketData:
    """VacGripPacketData"""

    # 원본 데이터 포맷: tick[1] + flag[1] + imu[7] + loadcell[1] + mic[32]
    DATA_FORMAT: str = (
        "L" + "?" + "hhhhhhh" + "H" + "H" * 32
    ) 

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
        # 출력 내용을 마이크로폰 데이터에 집중하도록 수정
        avg_mic = np.mean(self.mic)
        max_mic = np.max(self.mic)
        
        text = "T={:.4f}s | ".format(int(self.tick) / MCU_CLOCK)
        text += f"Flag:{self.flag} | MIC_Avg:{avg_mic:.2f} | MIC_Max:{max_mic}"
        return text


@dataclass
class VacGripRawData:
    """VacGripRawData (마이크로폰 전용)"""

    tick: npt.NDArray[np.uint32]
    flag: npt.NDArray[np.bool_] 
    # accelerometer: npt.NDArray[np.int16] # 제거됨
    # gyroscope: npt.NDArray[np.int16]    # 제거됨
    # temperature: npt.NDArray[np.int16]  # 제거됨
    # loadcell: npt.NDArray[np.uint16]    # 제거됨
    microphone: npt.NDArray[np.uint16]


@dataclass
class VacGripData:
    """VacGripData (마이크로폰 전용)"""

    time: npt.NDArray[np.float64] 
    flag: npt.NDArray[np.bool_] 
    # accelerometer: npt.NDArray[np.float64] # 제거됨
    # gyroscope: npt.NDArray[np.float64]    # 제거됨
    # temperature: npt.NDArray[np.float64]  # 제거됨
    # loadcell: npt.NDArray[np.float64]    # 제거됨
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
        0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x02F7, 0x02F2, \
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

    def _unpacking_if_valid(
        self, packet_candidate: bytes
    ) -> Optional[VacGripPacketData]:
        if len(packet_candidate) == self._packet_size:
            if packet_candidate[:4] == b"\xff\xff\xfd\x00":
                packetdata = VacGripPacketData(
                    struct.unpack(self._packet_format, packet_candidate)
                )
                if self._calculate_crc(packet_candidate[:-2]) == packetdata.crc:
                    return packetdata

        return None

    def generate_valid_packet(
        self, inputbuf: bytearray
    ) -> Iterator[Tuple[bytearray, VacGripPacketData]]:
        while self.packet_size <= len(inputbuf):
            packet_candidate = inputbuf[: self.packet_size]
            packetdata = self._unpacking_if_valid(packet_candidate)

            if packetdata is not None:
                del inputbuf[: self.packet_size]
                yield packet_candidate, packetdata

            else:
                del inputbuf[0] # 유효하지 않으면 한 바이트씩 버림

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
    # mic 데이터만 추출
    microphone = []

    packets = load_packets(file_path)
    if not packets:
         raise ValueError(f"파일이 비어있거나 읽을 수 없습니다: {file_path}")
         
    for _, packetdata in VacGripPacketHandler().generate_valid_packet(packets):
        tick.append(packetdata.tick)
        flag.append(packetdata.flag)
        microphone.append(packetdata.mic)

    return VacGripRawData(
        tick=np.array(tick, dtype=np.uint32),
        flag=np.array(flag, dtype=np.bool_),
        microphone=np.array(microphone, dtype=np.uint16), # 마이크 데이터는 uint16으로 유지
    )


def load_data(file_path: str) -> VacGripData:
    rawdata = load_rawdata(file_path)

    # 틱 오버플로우 처리를 위해 float64로 명시적 변환 (uint32 범위 초과 오류 방지)
    tick_float = rawdata.tick.astype(np.float64) 
    
    tick_dif = tick_float[1:] - tick_float[:-1]
    
    # 2**32 (4294967296.0)를 더하여 오버플로우 보정
    tick_dif[tick_dif < 0] = tick_dif[tick_dif < 0] + (2**32) 
    
    time_dif = tick_dif / MCU_CLOCK
    time = np.cumsum(time_dif)
    time = np.insert(time, 0, 0)

    # 마이크로폰 데이터만 반환
    return VacGripData(
        time=time,
        flag=rawdata.flag,
        microphone=rawdata.microphone / 16 / 4096, #  스케일링된 마이크 데이터
    )


class DataWriterState(Enum):
    """State"""

    IDLE = auto()
    COLLECTING = auto()
    FINISH_AND_SAVING = auto()
    SAVED = auto()


class DataWriter:
    """데이터 수집 및 저장을 담당하는 기본 클래스"""

    _timer: timer.Timer
    _state: DataWriterState
    _serial: serial.Serial
    _packet_handler: VacGripPacketHandler
    _inputbuf: bytearray
    _packets: bytearray
    _file_path: str

    def __init__(self, port_name: str, baud_rate: int = BAUD_RATE) -> None:
        self._timer = timer.Timer(0.001, self._update)
        try:
            self._serial = serial.Serial(port_name, baud_rate, timeout=0)
            self._serial.reset_input_buffer()
        except serial.SerialException as e:
            raise e
            
        self._packet_handler = VacGripPacketHandler()
        self._clear_buf()

        self._state = DataWriterState.IDLE
        self._timer.start()

    def close(self) -> None:
        if self._timer.is_running():
            self._timer.stop()
        if self._serial.is_open:
            self._serial.close()
            
    def is_running(self) -> bool:
        return self._timer.is_running()

    def _clear_buf(self) -> None:
        self._inputbuf = bytearray(0)
        self._packets = bytearray(0)

    def _update(self) -> None:
        # DataWriter는 COLLECTING 상태일 때만 데이터를 수집함 (출력 없음)
        if self._state == DataWriterState.COLLECTING:
            self._collect()
        elif self._state == DataWriterState.FINISH_AND_SAVING:
            self._save()
            self._clear_buf()
            self._state = DataWriterState.SAVED
            
    def _collect(self) -> None:
        packet_size = self._packet_handler.packet_size
        
        while True:
            new_bytes = self._serial.read(packet_size)
            self._inputbuf += new_bytes
            if len(new_bytes) < packet_size:
                break

        for packet, _ in self._packet_handler.generate_valid_packet(self._inputbuf):
            self._packets += packet

    def _save(self) -> None:
        packet_num = len(self._packets) / self._packet_handler.packet_size
        if packet_num == 0:
            print("No packets to save.")
        else:
            save_packets(self._file_path, self._packets)
            print(f'Saved {int(packet_num)} packets on "{self._file_path}".')

    def start(self) -> None:
        self._serial.write(b"s")
        self._state = DataWriterState.COLLECTING

    def stop(self) -> None:
        self._serial.write(b"f")
        self._state = DataWriterState.IDLE

    def finish_and_save(self, file_path: str) -> None:
        self._serial.write(b"f")
        time.sleep(0.1) 
        self._file_path = file_path
        self._state = DataWriterState.FINISH_AND_SAVING

    @property
    def state(self) -> DataWriterState:
        return self._state


class DataMonitor(DataWriter):
    """
    DataMonitor: DataWriter의 기능을 상속받아 수집과 마이크로폰 모니터링을 동시에 처리.
    COLLECTING 상태일 때만 print(packetdata)를 실행하여 모니터링을 제어함.
    """

    def _update(self) -> None:
        # COLLECTING 상태일 때만 수집(누적) 및 출력(모니터링)을 수행.
        if self._state == DataWriterState.COLLECTING:
            self._collect_and_monitor()
            
        elif self._state == DataWriterState.FINISH_AND_SAVING:
            self._save()
            self._clear_buf()
            self._state = DataWriterState.SAVED
    
    def _collect_and_monitor(self) -> None:
        packet_size = self._packet_handler.packet_size

        # 1. 원시 바이트 수집 
        while True:
            new_bytes = self._serial.read(packet_size)
            self._inputbuf += new_bytes
            if len(new_bytes) < packet_size:
                break

        # 2. 유효 패킷 추출 및 처리 (출력 + 누적)
        for packet_candidate, packetdata in self._packet_handler.generate_valid_packet(self._inputbuf):
            # 출력 (모니터링) - __str__에서 마이크 데이터만 출력하도록 수정됨
            print(packetdata) 
            
            # 누적 (수집)
            self._packets += packet_candidate