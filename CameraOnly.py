import time, numpy as np, cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
time.sleep(1)  # warm-up

# -----------------------------
# Model setup
# -----------------------------
MODEL_NAME = "yolov8n-tiny.pt"  # Tiny model for Pi CPU
model = YOLO(MODEL_NAME).to("cpu")

# -----------------------------
# Tracking setup
# -----------------------------
CONF_THR = 0.5
MID = 640 / 2
APPROACH_RATE, AWAY_RATE, MIN_MOVEMENT = 0.22, -0.22, 8
_next_id, tracks = 0, {}

def motion_label_trial(prev, curr):
    obj_oldpos1 = prev["x1"]
    obj_oldpos2 = prev["x2"]
    obj_newpos1 = curr["x1"]
    obj_newpos2 = curr["x2"]
    midpos_old = (obj_oldpos1 + obj_oldpos2) / 2
    midpos_new = (obj_newpos1 + obj_newpos2) / 2
    reldir = (abs(midpos_old - MID) - abs(midpos_new - MID))
    dir = "right" if midpos_old >= MID else "left"
    dir2 = "right" if midpos_new >= MID else "left"

    dt = max(curr["t"] - prev["t"], 1e-3)
    dx, dy = (curr["cx"] - prev["cx"]) / dt, (curr["cy"] - prev["cy"]) / dt
    if abs(dx) + abs(dy) < MIN_MOVEMENT:
        return "steady"
    elif reldir >= APPROACH_RATE:
        return f"approaching from the {dir}"
    elif reldir <= AWAY_RATE:
        return f"moving away to the {dir2}"
    else:
        return "steady"

# -----------------------------
# FIXED assign_ids() function
# -----------------------------
def assign_ids(objs):
    global _next_id, tracks
    assigned = {}
    old_ids = list(tracks.keys())

    # Handle empty lists
    if not objs:
        return assigned
    if not old_ids:
        for obj in objs:
            assigned[_next_id] = obj
            _next_id += 1
        return assigned

    # Compute distance matrix safely
    old = np.array([[tracks[i]["cx"], tracks[i]["cy"]] for i in old_ids], float)
    new = np.array([[o["cx"], o["cy"]] for o in objs], float)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1] != 2 or new.shape[1] != 2:
        for obj in objs:
            assigned[_next_id] = obj
            _next_id += 1
        return assigned

    dists = np.sqrt(((new[:, None, :] - old[None, :, :]) ** 2).sum(2))
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

# -----------------------------
# Draw + Track objects
# -----------------------------
def draw_and_track(frame, boxes, names):
    global tracks
    h, w = frame.shape[:2]
    now = time.monotonic()
    curr = []

    for b in boxes:
        if float(b.conf[0]) < CONF_THR:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(b.cls[0])
        score = float(b.conf[0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        area = ((x2 - x1) * (y2 - y1)) / float(w * h)
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
        state = motion_label_trial(prev, obj) if prev else "steady"

        last_said_t = prev.get("last_said_t", 0) if prev else 0
        if state.startswith("approaching") or state.startswith("moving away"):
            if now - last_said_t > 0.5:
                print(f"[{time.strftime('%H:%M:%S')}] Object {tid} {state} ({obj['label']})")
                obj["last_said_t"] = now
        else:
            obj["last_said_t"] = last_said_t

        obj["last_state"] = state
        new_tracks[tid] = obj

        # Draw bounding box
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj['label']} {obj['score']:.2f}",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.putText(frame, state, (x1, min(h - 5, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    tracks = new_tracks
    return frame

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        frame = picam2.capture_array()
        results = model(frame, verbose=False)
        for r in results:
            frame = draw_and_track(frame, r.boxes, model.names)

        # Save current frame for remote checking
        cv2.imwrite("/tmp/latest_frame.jpg", frame)

        # Press Ctrl+C to quit (no GUI used)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    picam2.close()
    print("✅ Finished.")



