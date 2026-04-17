import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# -----------------------
# CONFIG
# -----------------------
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

MIN_GREEN = 10
MAX_GREEN = 40
BASE_GREEN = 8
K_COUNT = 2
K_WAIT = 1.2

YELLOW = 3
FPS = 20

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/sample_traffic.mp4")

ret, frame = cap.read()
h, w = frame.shape[:2]

# -----------------------
# 4 REGIONS (N, S, E, W)
# -----------------------
regions = {
    "north": (0, 0, w, h // 2),
    "south": (0, h // 2, w, h),
    "west":  (0, 0, w // 2, h),
    "east":  (w // 2, 0, w, h),
}

def get_region_name(x, y):
    if y < h // 2:
        return "north"
    elif y >= h // 2:
        return "south"
    if x < w // 2:
        return "west"
    else:
        return "east"

# -----------------------
# QUEUE + WAIT TIME
# -----------------------
history_len = FPS * 10
queues = {r: deque(maxlen=history_len) for r in regions}
waiting_time = {r: 0 for r in regions}

approaches = list(regions.keys())
current_idx = 0

def compute_green_time(count, wait):
    return int(max(MIN_GREEN,
                   min(MAX_GREEN,
                       BASE_GREEN + K_COUNT * count + K_WAIT * wait)))

# -----------------------
# MAIN LOOP
# -----------------------
while cap.isOpened():

    cur_approach = approaches[current_idx]

    latest_count = sum(queues[cur_approach]) if queues[cur_approach] else 0
    wait = waiting_time[cur_approach]

    green_time = compute_green_time(latest_count, wait)

    start_time = time.time()
    end_time = start_time + green_time

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        results = model(frame, device='cpu')

        boxes = results[0].boxes
        names = results[0].names

        counts = {r: 0 for r in regions}

        for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
            class_name = names[int(cls)]

            if class_name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                region = get_region_name(cx, cy)
                if region in counts:
                    counts[region] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Update queues
        for r in queues:
            queues[r].append(counts[r])

        # Update waiting time
        for r in waiting_time:
            if r != cur_approach:
                waiting_time[r] += 1
            else:
                waiting_time[r] = 0

        # Display signals
        for i, a in enumerate(approaches):
            color = (0, 255, 0) if a == cur_approach else (0, 0, 255)

            cv2.putText(frame, f"{a.upper()} [{'GREEN' if a == cur_approach else 'RED'}]",
                        (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            qlen = int(np.mean(queues[a])) if queues[a] else 0

            cv2.putText(frame, f"Q:{qlen} W:{waiting_time[a]}",
                        (10, 50 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Timer
        rem = int(end_time - time.time())
        cv2.putText(frame, f"Time Left: {rem}s",
                    (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Smart Traffic System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # YELLOW PHASE
    yellow_start = time.time()
    while time.time() < yellow_start + YELLOW:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        cv2.putText(frame, f"{cur_approach.upper()} YELLOW",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Smart Traffic System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # Next signal
    current_idx = (current_idx + 1) % len(approaches)

cap.release()
cv2.destroyAllWindows()