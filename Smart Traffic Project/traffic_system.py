import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt


MODEL_NAME = "yolov8n.pt"   
VEHICLE_CLASSES = {"car","truck","bus","motorcycle"}  # classes to count
MIN_GREEN = 10   # seconds (minimum green)
MAX_GREEN = 40   # seconds (maximum green)
BASE_GREEN = 10  # base green seconds
K = 2            # multiplier (seconds per detected vehicle)
YELLOW = 3       # yellow time in seconds
FPS = 20         # approximate frames per second of input


model = YOLO(MODEL_NAME)


VIDEO_SOURCE = "videos/sample_traffic.mp4"  # change to 0 for webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)


ret, frame = cap.read()
h, w = frame.shape[:2]

# Regions (top, right, bottom, left)
regions = {
    "north":  (0, 0, w, h//2),
    "south":  (0, h//2, w, h),
    # "east":  (w//2, 0, w, h),
    # "west":  (0, 0, w//2, h)
}


# -----------------------
# Utility functions
# -----------------------
def get_region_name(x, y):
    # x,y are center coordinates of box
    # return 'north' or 'south' based on y
    return "north" if y < h//2 else "south"

# Running queue counts (smooth using deque)
history_len = FPS * 10  # keep 10 seconds history
queues = {r: deque(maxlen=history_len) for r in regions.keys()}

# Logging
log = []

# Traffic controller state
approaches = list(regions.keys())
current_idx = 0

def compute_green_time(count):
    # Simple proportional rule
    return int(max(MIN_GREEN, min(MAX_GREEN, BASE_GREEN + K * count)))


# Start with current approach green
while cap.isOpened():
    # Control cycle for current approach
    cur_approach = approaches[current_idx]
    # compute current count as smoothed average
    cur_count = int(np.mean([sum(queues[a]) if queues[a] else 0 for a in queues]))
    # But better: use latest observed count for current approach
    latest_count = sum(queues[cur_approach]) if queues[cur_approach] else 0
    green_time = compute_green_time(latest_count)
    start_time = time.time()
    end_time = start_time + green_time

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            ret, frame = cap.read()
        # Run detection (Ultralytics returns results)
        results = model(frame, device='cpu')  # use 'cuda:0' if GPU available

        # parse results: results[0].boxes.xyxy, .cls, .conf
        boxes = results[0].boxes
        names = results[0].names

        # Count vehicles per region this frame
        counts = {r:0 for r in regions.keys()}
        for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
            class_name = names[int(cls)]
            if class_name in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, box[:4])
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                region = get_region_name(cx, cy)
                if region in counts:
                    counts[region] += 1
                    # draw box
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame, class_name, (x1,y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        # Update queues: append counts for smoothing
        for r in queues:
            queues[r].append(counts.get(r,0))

        # Overlay signal info
        for i, a in enumerate(approaches):
            color = (0,255,0) if a==cur_approach else (0,0,255)
            cv2.putText(frame, f"{a} [{'GREEN' if a==cur_approach else 'RED'}]", (10,30+30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # show smoothed queue length
            qlen = int(np.mean(queues[a])) if queues[a] else 0
            cv2.putText(frame, f"Queue: {qlen}", (10,55+30*i), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

        # show timer
        rem = int(end_time - time.time())
        cv2.putText(frame, f"Green time left: {rem}s", (w-300,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.imshow("Traffic Control Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            # optional: save log/plots here
            exit(0)

    # Yellow period
    yellow_start = time.time()
    while time.time() < yellow_start + YELLOW:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cv2.putText(frame, f"{cur_approach} YELLOW", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.imshow("Traffic Control Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # Move to next approach
    current_idx = (current_idx + 1) % len(approaches)

# release
cap.release()
cv2.destroyAllWindows()
