import signal
import serial
from gpiozero import DistanceSensor
import time, numpy as np, cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO


# ============================================================
# GPIO / Buzzer setup
# ============================================================

BuzzR = 17
BuzzL = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(BuzzR, GPIO.OUT)
GPIO.setup(BuzzL, GPIO.OUT)

pwmr = GPIO.PWM(BuzzR, 100)
pwml = GPIO.PWM(BuzzL, 100)

ahh = False
nah = False
s = ""


# ============================================================
# SEN0646 TOF UART setup
# ============================================================

TOF_PORT = "/dev/serial0"
TOF_BAUD = 921600

tof_serial = serial.Serial(
    port=TOF_PORT,
    baudrate=TOF_BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0
)

tof_buffer = bytearray()
latest_distance_mm = None
latest_distance_cm = None


def tof_checksum_ok(frame):
    """
    SEN0646 / NLink checksum:
    checksum = low 8 bits of sum of bytes 0..14
    """
    return (sum(frame[:15]) & 0xFF) == frame[15]


def decode_tof_frame(frame):
    """
    Decode a 16-byte NLink_TOFSense_Frame0 frame.

    Frame:
        0   = 0x57
        1   = 0x00
        2   = reserved
        3   = sensor ID
        4-7 = system time
        8-10 = distance, 24-bit little endian, mm
        11  = distance status
        12-13 = signal strength
        14  = ranging precision
        15  = checksum
    """

    if len(frame) != 16:
        return None

    if frame[0] != 0x57:
        return None

    if frame[1] != 0x00:
        return None

    if not tof_checksum_ok(frame):
        return None

    sensor_id = frame[3]

    system_time_ms = int.from_bytes(
        frame[4:8],
        byteorder="little"
    )

    # 24-bit little-endian distance in mm
    distance_mm = (
        frame[8]
        | (frame[9] << 8)
        | (frame[10] << 16)
    )

    distance_status = frame[11]

    signal_strength = int.from_bytes(
        frame[12:14],
        byteorder="little"
    )

    range_precision_cm = frame[14]

    return {
        "id": sensor_id,
        "system_time_ms": system_time_ms,
        "distance_mm": distance_mm,
        "distance_cm": distance_mm / 10.0,
        "distance_m": distance_mm / 1000.0,
        "distance_status": distance_status,
        "signal_strength": signal_strength,
        "range_precision_cm": range_precision_cm,
    }


def update_tof():
    """
    Read all currently available UART data and extract
    the newest valid SEN0646 frame.

    Non-blocking: returns immediately if no new data exists.
    """

    global tof_buffer
    global latest_distance_mm
    global latest_distance_cm

    try:
        waiting = tof_serial.in_waiting

        if waiting > 0:
            tof_buffer.extend(tof_serial.read(waiting))

    except serial.SerialException as e:
        print(f"TOF UART error: {e}")
        return

    # Search for complete frames
    while len(tof_buffer) >= 16:

        # Find the header byte 0x57
        try:
            header_index = tof_buffer.index(0x57)
        except ValueError:
            # No possible frame header
            tof_buffer.clear()
            return

        # Remove garbage before header
        if header_index > 0:
            del tof_buffer[:header_index]

        # Still not enough data for a complete frame
        if len(tof_buffer) < 16:
            return

        frame = bytes(tof_buffer[:16])

        decoded = decode_tof_frame(frame)

        if decoded is not None:
            latest_distance_mm = decoded["distance_mm"]
            latest_distance_cm = decoded["distance_cm"]

            # Remove processed frame
            del tof_buffer[:16]

        else:
            # Bad frame.
            # Remove only the first byte and search for another
            # possible 0x57 header.
            del tof_buffer[0]


# ============================================================
# Camera setup
# ============================================================

picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)

picam2.start()

time.sleep(1)  # warm-up


# ============================================================
# Model setup
# ============================================================

MODEL_NAME = "yolov8n.pt"

model = YOLO(MODEL_NAME).to("cpu")


# ============================================================
# Tracking setup
# ============================================================

CONF_THR = 0.5
MID = 640 / 2

APPROACH_RATE = 0.22
AWAY_RATE = -0.22
MIN_MOVEMENT = 8

_next_id = 0
tracks = {}


def motion_label_trial(prev, curr):

    obj_oldpos1 = prev["x1"]
    obj_oldpos2 = prev["x2"]

    obj_newpos1 = curr["x1"]
    obj_newpos2 = curr["x2"]

    midpos_old = (obj_oldpos1 + obj_oldpos2) / 2
    midpos_new = (obj_newpos1 + obj_newpos2) / 2

    reldir = (
        abs(midpos_old - MID)
        - abs(midpos_new - MID)
    )

    dir1 = "right" if midpos_old >= MID else "left"
    dir2 = "right" if midpos_new >= MID else "left"

    dt = max(curr["t"] - prev["t"], 1e-3)

    dx = (curr["cx"] - prev["cx"]) / dt
    dy = (curr["cy"] - prev["cy"]) / dt

    if abs(dx) + abs(dy) < MIN_MOVEMENT:
        return "steady"

    elif reldir >= APPROACH_RATE:
        return f"approaching from the {dir1}"

    elif reldir <= AWAY_RATE:
        return f"moving away to the {dir2}"

    else:
        return "steady"


# ============================================================
# assign_ids()
# ============================================================

def assign_ids(objs):

    global _next_id, tracks

    assigned = {}

    old_ids = list(tracks.keys())

    if not objs:
        return assigned

    if not old_ids:

        for obj in objs:
            assigned[_next_id] = obj
            _next_id += 1

        return assigned

    old = np.array(
        [
            [tracks[i]["cx"], tracks[i]["cy"]]
            for i in old_ids
        ],
        float
    )

    new = np.array(
        [
            [o["cx"], o["cy"]]
            for o in objs
        ],
        float
    )

    if (
        old.ndim != 2
        or new.ndim != 2
        or old.shape[1] != 2
        or new.shape[1] != 2
    ):

        for obj in objs:
            assigned[_next_id] = obj
            _next_id += 1

        return assigned

    dists = np.sqrt(
        (
            (
                new[:, None, :]
                - old[None, :, :]
            ) ** 2
        ).sum(2)
    )

    used = set()

    for ni, obj in enumerate(objs):

        mid = None

        if dists.size:

            oi = int(np.argmin(dists[ni]))

            if oi not in used:

                mid = old_ids[oi]
                used.add(oi)

        if mid is None:

            mid = _next_id
            _next_id += 1

        assigned[mid] = obj

    return assigned


# ============================================================
# Draw + Track objects
# ============================================================

def draw_and_track(frame, boxes, names):

    global tracks

    h, w = frame.shape[:2]

    now = time.monotonic()

    curr = []

    for b in boxes:

        if float(b.conf[0]) < CONF_THR:
            continue

        x1, y1, x2, y2 = map(
            int,
            b.xyxy[0].tolist()
        )

        if x2 <= x1 or y2 <= y1:
            continue

        cls = int(b.cls[0])

        score = float(b.conf[0])

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        area = (
            (x2 - x1) * (y2 - y1)
        ) / float(w * h)

        curr.append(
            {
                "bbox": (x1, y1, x2, y2),
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "area": area,
                "t": now,
                "label": names[cls],
                "score": score,
            }
        )

    if not curr:

        print("No detections.")
        return frame

    assigned = assign_ids(curr)

    new_tracks = {}

    for tid, obj in assigned.items():

        prev = tracks.get(tid)

        state = (
            motion_label_trial(prev, obj)
            if prev
            else "steady"
        )

        # ------------------------------------------------
        # RIGHT approaching
        # ------------------------------------------------

        if state == "approaching from the right":

            frequency = (
                500
                + obj["area"] * 3500
            )

            pwmr.ChangeFrequency(frequency)

            pwmr.start(50)

            time.sleep(0.5)

            if ahh is False:
                pwmr.stop()

        # ------------------------------------------------
        # LEFT approaching
        # ------------------------------------------------

        if state == "approaching from the left":

            frequency = (
                500
                + obj["area"] * 3500
            )

            pwml.ChangeFrequency(frequency)

            pwml.start(50)

            time.sleep(0.5)

            if ahh is False:
                pwml.stop()

        last_said_t = (
            prev.get("last_said_t", 0)
            if prev
            else 0
        )

        if (
            state.startswith("approaching")
            or state.startswith("moving away")
        ):

            if now - last_said_t > 0.5:

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Object {tid} "
                    f"{state} "
                    f"({obj['label']})"
                )

                obj["last_said_t"] = now

        else:

            obj["last_said_t"] = last_said_t

        obj["last_state"] = state

        new_tracks[tid] = obj

        # ------------------------------------------------
        # Draw bounding box
        # ------------------------------------------------

        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{obj['label']} {obj['score']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            state,
            (x1, min(h - 5, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    tracks = new_tracks

    return frame


# ============================================================
# Main loop
# ============================================================

try:

    while True:

        # ====================================================
        # Update SEN0646 UART
        # ====================================================

        update_tof()

        # Use most recent valid TOF measurement
        if latest_distance_cm is not None:

            distance_cm = latest_distance_cm

            # Store/print only when value changes meaningfully
            rounded_distance = round(distance_cm, 1)

            if s != rounded_distance:

                s = rounded_distance

                print(
                    f"TOF distance: "
                    f"{distance_cm:.1f} cm"
                )

        else:

            # No TOF frame received yet
            distance_cm = 999.0

        # ====================================================
        # TOF safety threshold
        # ====================================================

        if distance_cm <= 200:

            ahh = True

        else:

            ahh = False

        # ====================================================
        # Warning buzzers
        # ====================================================

        if ahh is True:

            pwmr.ChangeFrequency(1500)
            pwml.ChangeFrequency(1500)

            pwmr.start(50)
            pwml.start(50)

            time.sleep(0.1)

            pwmr.stop()
            pwml.stop()

            time.sleep(0.1)

            pwmr.ChangeFrequency(1500)
            pwml.ChangeFrequency(1500)

            pwmr.start(50)
            pwml.start(50)

            time.sleep(0.1)

            pwmr.stop()
            pwml.stop()

        else:

            pwmr.stop()
            pwml.stop()

        # ====================================================
        # Camera
        # ====================================================

        frame = picam2.capture_array()

        results = model(
            frame,
            verbose=False
        )

        for r in results:

            frame = draw_and_track(
                frame,
                r.boxes,
                model.names
            )

        # ====================================================
        # Save latest frame
        # ====================================================

        cv2.imwrite(
            "/tmp/latest_frame.jpg",
            frame
        )

        # ====================================================
        # Small delay
        # ====================================================

        time.sleep(0.05)


except KeyboardInterrupt:

    print("🛑 Stopped by user.")


finally:

    try:
        tof_serial.close()
    except Exception:
        pass

    try:
        pwmr.stop()
        pwml.stop()
    except Exception:
        pass

    picam2.close()

    GPIO.cleanup()

    print("✅ Finished.")

