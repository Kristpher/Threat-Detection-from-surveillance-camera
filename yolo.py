import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_DIR  = "/workspace/CMeRT/testCM"
OUTPUT_DIR = "/workspace/CMeRT/ttc/results_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THREAT_CLASSES = [0, 1, 2, 3, 5, 7]

TTC_MIN = 0.1
TTC_MAX = 30.0

model = YOLO("yolov8n.pt")
print(f"Model loaded. Processing videos in {VIDEO_DIR}")
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
print(f"Found {len(video_files)} videos\n")

for video_file in tqdm(video_files, desc="Videos"):
    stem      = video_file.replace(".mp4", "")
    out_path  = os.path.join(OUTPUT_DIR, stem + "_ttc.npy")
    if os.path.exists(out_path):
        print(f"  [SKIP] {stem} already processed")
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_centers  = {}
    ttc_per_frame = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=THREAT_CLASSES,
            verbose=False,
            tracker="bytetrack.yaml"
        )[0]

        centers = {}
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids   = results.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box[:4]
                centers[tid] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        min_ttc = TTC_MAX
        if len(centers) >= 2:
            tids = list(centers.keys())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    ti, tj = tids[i], tids[j]
                    if ti not in prev_centers or tj not in prev_centers:
                        continue
                    rel_pos   = centers[tj] - centers[ti]
                    vel_i     = centers[ti] - prev_centers[ti]
                    vel_j     = centers[tj] - prev_centers[tj]
                    rel_vel   = vel_j - vel_i
                    distance  = np.linalg.norm(rel_pos)
                    rel_speed = np.linalg.norm(rel_vel)
                    if rel_speed > 1e-3:
                        ttc = distance / (rel_speed * fps)
                        ttc = np.clip(ttc, TTC_MIN, TTC_MAX)
                        if ttc < min_ttc:
                            min_ttc = ttc

        ttc_per_frame.append(min_ttc)
        prev_centers = centers.copy()

    cap.release()
    ttc_array = np.array(ttc_per_frame, dtype=np.float32)
    np.save(out_path, ttc_array)
    print(f"  [DONE] {stem}  frames={len(ttc_array)}  min_ttc={ttc_array.min():.2f}  mean_ttc={ttc_array.mean():.2f}")

print(f"\nAll done. Saved to {OUTPUT_DIR}")
