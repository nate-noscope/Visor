#!/usr/bin/env python3

import serial
import time


PORT = "/dev/serial0"
BAUD = 921600

FRAME_HEADER = 0x57
FUNCTION_MARK = 0x00
FRAME_LENGTH = 16


def checksum_ok(frame):
    """Verify NLink checksum."""
    return (sum(frame[:15]) & 0xFF) == frame[15]


def decode_frame(frame):
    """
    Decode a 16-byte NLink_TOFSense_Frame0 frame.

    Returns a dictionary or None if invalid.
    """

    if len(frame) != FRAME_LENGTH:
        return None

    if frame[0] != FRAME_HEADER:
        return None

    if frame[1] != FUNCTION_MARK:
        return None

    if not checksum_ok(frame):
        return None

    # Byte 2: reserved
    sensor_id = frame[3]

    # Bytes 4-7: system time, little endian
    system_time_ms = int.from_bytes(
        frame[4:8],
        byteorder="little"
    )

    # Bytes 8-10: unsigned 24-bit distance, millimeters
    distance_mm = (
        frame[8]
        | (frame[9] << 8)
        | (frame[10] << 16)
    )

    distance_m = distance_mm / 1000.0

    # Byte 11
    distance_status = frame[11]

    # Bytes 12-13: signal strength, little endian
    signal_strength = int.from_bytes(
        frame[12:14],
        byteorder="little"
    )

    # Byte 14: reference ranging precision, cm
    range_precision_cm = frame[14]

    return {
        "id": sensor_id,
        "system_time_ms": system_time_ms,
        "distance_mm": distance_mm,
        "distance_m": distance_m,
        "distance_status": distance_status,
        "signal_strength": signal_strength,
        "range_precision_cm": range_precision_cm,
    }


def read_tof(ser):
    """
    Find and decode the next valid 16-byte frame.
    """

    while True:
        # Search for 0x57
        byte = ser.read(1)

        if not byte:
            return None

        if byte[0] != FRAME_HEADER:
            continue

        # Read remaining 15 bytes
        rest = ser.read(FRAME_LENGTH - 1)

        if len(rest) != FRAME_LENGTH - 1:
            return None

        frame = byte + rest

        decoded = decode_frame(frame)

        if decoded is not None:
            return decoded

        # Invalid frame.
        # Loop back and search for the next 0x57.


def main():

    print(f"Opening {PORT} at {BAUD} baud...")

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    # Discard anything already sitting in the UART buffer
    ser.reset_input_buffer()

    print("Waiting for TOF measurements...\n")

    try:

        while True:

            data = read_tof(ser)

            if data is None:
                continue

            print(
                f"Distance: {data['distance_m']:.3f} m  "
                f"({data['distance_mm']} mm) | "
                f"Status: {data['distance_status']} | "
                f"Signal: {data['signal_strength']} | "
                f"Precision: ±{data['range_precision_cm']} cm"
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        ser.close()


if __name__ == "__main__":
    main()


