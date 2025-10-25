# yolo_advanced.py
# Copyright (c) 2025 Shivam
# Based on public examples and Ultralytics YOLO.
# MIT-style short notice: modify/use freely; keep a short credit in distributions.
#
# Requirements:
#   pip install ultralytics opencv-python numpy
#
# Controls (while window focused):
#   q -> quit
#   s -> toggle saving output file
#   p -> pause/unpause
#   + -> increase pixel_per_meter (makes reported distances smaller)
#   - -> decrease pixel_per_meter (makes reported distances larger)
#   c -> clear tracker
#   h -> toggle help overlay

import time
import math
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"   # or yolov8s.pt etc.
USE_WEBCAM = True          # if True uses cv2.VideoCapture(0); if False, uses CAMERA_URL
CAMERA_URL = "http://<Your_IP_address>:<Your_Port>/video"
OUTPUT_FILE = "output_advanced.avi"
INITIAL_PIXEL_PER_METER = 234.0  
DISTANCE_SMOOTH_WINDOW = 5       
TRACKER_MAX_DISTANCE = 80        
FPS_SMOOTH = 5                   
# ----------------------------------------

# Lightweight centroid tracker
class SimpleTracker:
    def __init__(self, max_distance=TRACKER_MAX_DISTANCE):
        self.next_id = 1
        self.objects = {}  # id -> (cx, cy)
        self.history = {}  # id -> deque of last distances
        self.max_distance = max_distance

    def reset(self):
        self.next_id = 1
        self.objects.clear()
        self.history.clear()

    def update(self, centroids):
        """
        centroids: list of (cx, cy, distance_m, cls_idx, conf)
        returns: list of (id, cx, cy, distance_m, cls_idx, conf)
        """
        assigned = {}
        results = []

        if len(self.objects) == 0:
            # create all
            for c in centroids:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = (c[0], c[1])
                dq = deque(maxlen=DISTANCE_SMOOTH_WINDOW)
                dq.append(c[2])
                self.history[oid] = dq
                results.append((oid, c[0], c[1], c[2], c[3], c[4]))
            return results

        # build distance matrix between existing objects and new centroids
        existing_ids = list(self.objects.keys())
        existing_pts = np.array([self.objects[i] for i in existing_ids], dtype=np.float32)
        new_pts = np.array([[c[0], c[1]] for c in centroids], dtype=np.float32) if len(centroids) else np.empty((0,2), dtype=np.float32)

        if new_pts.shape[0] == 0:
            # no detections: optionally keep objects but return nothing
            return []

        distances = np.linalg.norm(existing_pts[:, None, :] - new_pts[None, :, :], axis=2)  # shape (E, N)

        # Greedy matching: smallest distance pairs first
        E, N = distances.shape
        used_existing = set()
        used_new = set()
        pairs = []
        for _ in range(min(E, N)):
            # Compatibility-safe way to find argmin ignoring inf entries:
            # convert inf to nan and use nanargmin.
            if np.all(np.isinf(distances)):
                break
            try:
                idx_flat = np.nanargmin(np.where(np.isinf(distances), np.nan, distances))
                idx = np.unravel_index(idx_flat, distances.shape)
            except ValueError:
                # all entries are NaN (no valid matches)
                break
            ei, ni = idx
            dist_val = distances[ei, ni]
            if np.isinf(dist_val) or np.isnan(dist_val):
                # nothing valid left
                break
            if dist_val > self.max_distance:
                # too far; mark this pair invalid and continue
                distances[ei, :] = np.inf
                distances[:, ni] = np.inf
                continue
            pairs.append((ei, ni, float(dist_val)))
            distances[ei, :] = np.inf
            distances[:, ni] = np.inf

        # assign matched
        for ei, ni, dval in pairs:
            oid = existing_ids[ei]
            c = centroids[ni]
            assigned[ni] = oid
            self.objects[oid] = (c[0], c[1])
            # ensure history exists
            if oid not in self.history:
                self.history[oid] = deque(maxlen=DISTANCE_SMOOTH_WINDOW)
            self.history[oid].append(c[2])
            avg_dist = float(np.mean(self.history[oid]))
            results.append((oid, c[0], c[1], avg_dist, c[3], c[4]))
            used_existing.add(ei)
            used_new.add(ni)

        # remaining new detections -> new IDs
        for ni, c in enumerate(centroids):
            if ni in used_new:
                continue
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = (c[0], c[1])
            dq = deque(maxlen=DISTANCE_SMOOTH_WINDOW)
            dq.append(c[2])
            self.history[oid] = dq
            results.append((oid, c[0], c[1], float(c[2]), c[3], c[4]))

        # Optionally: remove existing objects that were not updated for long (not implemented here)
        return results

# Utility for overlay text
def draw_overlay(frame, text_lines, origin=(10, 20), line_height=18, bg_alpha=0.6):
    # Draw semi-transparent background
    overlay = frame.copy()
    h = line_height * len(text_lines) + 8
    cv2.rectangle(overlay, (origin[0]-6, origin[1]-line_height), (origin[0]+500, origin[1]+h), (0,0,0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1-bg_alpha, 0, frame)
    y = origin[1]
    for line in text_lines:
        cv2.putText(frame, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y += line_height

# Main
def main():
    pixel_per_meter = float(INITIAL_PIXEL_PER_METER)
    saving = True
    paused = False
    show_help = True

    # Load model (will download weights if not present)
    model = YOLO(MODEL_PATH)

    # open capture
    cap = cv2.VideoCapture(0 if USE_WEBCAM else CAMERA_URL)
    if not cap.isOpened():
        print("Error opening video capture. Check CAMERA_URL or webcam.")
        return

    # --- read one initial frame to get true width/height ---
    ret_init, frame_init = cap.read()
    if not ret_init:
        print("Error: cannot read initial frame from camera.")
        cap.release()
        return

    h, w = frame_init.shape[:2]
    fps_camera = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fps_camera = fps_camera if fps_camera > 0 else 20.0

    # video writer (create after we know w,h)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    if saving:
        try:
            out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps_camera, (w, h))
        except Exception as e:
            print("Warning: cannot create VideoWriter:", e)
            out = None

    tracker = SimpleTracker()
    fps_deque = deque(maxlen=FPS_SMOOTH)
    last_time = time.time()

    window_name = "YOLOv8 Advanced - press 'h' for help"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    print("Starting. Press 'q' to quit, 'h' for help overlay.")

    # Use the already-read frame on the first loop iteration
    first_frame = frame_init.copy()
    first_frame_used = False

    while True:
        if not paused:
            # Use cached first frame once to have correct shape and avoid size mismatch
            if not first_frame_used:
                frame = first_frame
                first_frame_used = True
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Frame not grabbed — breaking.")
                    break

            # inference
            results = model(frame)  # returns list of Results (usually 1)
            if len(results) == 0:
                res = None
            else:
                res = results[0]

            centroids = []
            class_counts = {}
            annotator = Annotator(frame, line_width=2, example=str((w//2, h)))

            if res is not None and hasattr(res, 'boxes') and len(res.boxes) > 0:
                num_boxes = len(res.boxes)
                for i in range(num_boxes):
                    # get xyxy
                    xyxy = res.boxes.xyxy[i].cpu().numpy()  # numpy array [x1,y1,x2,y2]
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    # class & conf
                    cls_tensor = res.boxes.cls[i]
                    cls_idx = int(cls_tensor.cpu().numpy()) if hasattr(cls_tensor, 'cpu') else int(cls_tensor)
                    conf_tensor = res.boxes.conf[i] if hasattr(res.boxes, 'conf') else None
                    conf = float(conf_tensor.cpu().numpy()) if conf_tensor is not None and hasattr(conf_tensor, 'cpu') else 0.0

                    label = model.names.get(cls_idx, str(cls_idx)) if hasattr(model, 'names') else str(cls_idx)
                    class_counts[label] = class_counts.get(label, 0) + 1

                    # centroid
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # distance calc (pixels to meters using bottom-center reference)
                    center_point = (w // 2, h)
                    dist_px = math.hypot(cx - center_point[0], cy - center_point[1])
                    distance_m = dist_px / pixel_per_meter

                    centroids.append((cx, cy, distance_m, cls_idx, conf))

                    # draw raw bbox with label and confidence
                    box_color = colors(cls_idx, True)
                    annotator.box_label((x1, y1, x2, y2), f"{label} {conf:.2f}", color=box_color)

                # tracking update
                tracked = tracker.update(centroids)
            else:
                tracked = []

            # Draw tracking results (IDs, smoothed distance)
            for oid, cx, cy, dist_m, cls_idx, conf in tracked:
                # draw ID circle and text
                txt = f"ID:{oid} {dist_m:.2f}m"
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(frame, txt, (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

                # optional: draw a small line to bottom center
                cv2.line(frame, (w//2, h), (int(cx), int(cy)), (255, 0, 255), 1)

            # finalize annotator overlay
            out_frame = annotator.result()

            # compute FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_deque.append(fps)
            fps_smoothed = sum(fps_deque) / len(fps_deque)

            # build overlay text
            info_lines = [
                f"FPS: {fps_smoothed:.1f}  |  Saving: {'ON' if saving and out is not None else 'OFF'}  |  Pixel/m: {pixel_per_meter:.1f}",
                f"Tracked objects: {len(tracked)}"
            ]
            # show per-class counts
            if len(class_counts) > 0:
                counts = "  ".join([f"{k}:{v}" for k,v in class_counts.items()])
                info_lines.append("Counts: " + counts)

            if show_help:
                info_lines += [
                    "Controls: q=quit  s=toggle save  p=pause  +/- adjust pixel_per_meter",
                    "c=clear tracker  h=toggle help"
                ]

            draw_overlay(out_frame, info_lines, origin=(10, 30))

            cv2.imshow(window_name, out_frame)

            # write if saving
            if saving and out is not None:
                out.write(out_frame)

        # key handling (outside paused block so toggles still work)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving
            if saving and out is None:
                try:
                    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps_camera, (w, h))
                    print("Started saving to", OUTPUT_FILE)
                except Exception as e:
                    print("Cannot start saving:", e)
                    out = None
            elif not saving and out is not None:
                out.release()
                out = None
                print("Stopped saving.")
        elif key == ord('p'):
            paused = not paused
            print("Paused:", paused)
        elif key == ord('+') or key == ord('='):
            pixel_per_meter *= 1.05
        elif key == ord('-') or key == ord('_'):
            pixel_per_meter /= 1.05
        elif key == ord('c'):
            tracker.reset()
            print("Tracker reset.")
        elif key == ord('h'):
            show_help = not show_help
        # small sleep when paused to reduce CPU
        if paused:
            time.sleep(0.05)

    # cleanup
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
