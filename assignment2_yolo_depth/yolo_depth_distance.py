"""Project 2: YOLOv8 + Depth Anything V2 -- metric distance between two objects
from a single monocular camera.

Follows the steps described on the AI4DM page:
  Frame I   detect drone + target with YOLO      -> bounding boxes
  Frame II  project anchors onto the depth map   -> Z_drone, Z_target
  Frame III convert to camera coords, take the   -> D via law of cosines
            angle between the two rays

Example:
    python yolo_depth_distance.py --source input.mp4 --drone-class 4 --target-class 0
    python yolo_depth_distance.py --source 0 --show          # live webcam
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# The relative model gives back-to-front depth with no units (bigger means closer).
# Only the Metric models give real metres, which the distance maths needs.
METRIC_MODELS = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "relative": "depth-anything/Depth-Anything-V2-Small-hf",
}


def bbox_median_depth(depth: np.ndarray, x1, y1, x2, y2, window: int = 5):
    """Median depth in a small central window of a box, falling back to the whole box."""
    h, w = depth.shape
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    if window > 1:
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, window // 2
        central = depth[max(0, cy - half) : min(h, cy + half + 1), max(0, cx - half) : min(w, cx + half + 1)]
        if central.size:
            return float(np.median(central))
    roi = depth[y1 : y2 + 1, x1 : x2 + 1]
    return float(np.median(roi)) if roi.size else None


def to_camera_coords(u, v, z, K):
    """Back-project an image point (u, v) at depth z into camera coordinates."""
    fx, fy, cx, cy = K
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def distance_law_of_cosines(a, b):
    """The formula from the page. Only uses left-right angle, so it ignores height."""
    xa, _, za = a
    xb, _, zb = b
    dtheta = math.atan2(xb, zb) - math.atan2(xa, za)
    return math.sqrt(max(0.0, za**2 + zb**2 - 2 * za * zb * math.cos(dtheta)))


def distance_euclidean_3d(a, b):
    """Normal 3D distance, which also takes height into account."""
    return math.dist(a, b)


def default_intrinsics(width, height, hfov_deg):
    """Approximate pinhole intrinsics from a horizontal field of view (use real calibration!)."""
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return (fx, fx, width / 2, height / 2)


def get_depth(depth_pipe, frame_bgr, width, height):
    """Return a float32 metric depth map at the frame resolution."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    out = depth_pipe(pil)
    # For a single image this gives back a dictionary, not a list, so the page's [0]
    # crashes. Also out["depth"] is just a picture of the depth, rescaled every frame.
    # The real numbers are in out["predicted_depth"].
    if isinstance(out, list):
        out = out[0]
    depth = out["predicted_depth"] if isinstance(out, dict) and "predicted_depth" in out else out["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HERE / "input.mp4"), help="video path, or a camera index like 0")
    ap.add_argument("--out-video", default=str(OUT / "output_with_depth.mp4"))
    ap.add_argument("--out-csv", default=str(OUT / "detections_with_depth.csv"))
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--depth-mode", default="indoor", choices=list(METRIC_MODELS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--skip", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--device", default=None, help="cuda index or 'cpu' (auto by default)")
    ap.add_argument("--drone-class", type=int, default=None, help="COCO class id acting as the 'drone'")
    ap.add_argument("--target-class", type=int, default=None, help="COCO class id acting as the 'target'")
    ap.add_argument("--hfov", type=float, default=70.0, help="horizontal FOV in degrees, if not calibrated")
    ap.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"), default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    Path(a.out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)

    device = a.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    elif device != "cpu":
        device = int(device)

    print(f"Device: {device}")
    yolo = YOLO(a.weights)
    depth_pipe = pipeline(task="depth-estimation", model=METRIC_MODELS[a.depth_mode], device=device)
    if a.depth_mode == "relative":
        print("WARNING: 'relative' depth is unitless inverse depth -- distances will NOT be in metres.")

    source = int(a.source) if str(a.source).isdigit() else a.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {a.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    K = tuple(a.intrinsics) if a.intrinsics else default_intrinsics(width, height, a.hfov)
    print(f"Video {width}x{height} @ {fps:.1f} FPS, {total or '?'} frames | fx={K[0]:.1f} cx={K[2]:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(a.out_video, fourcc, fps / max(1, a.skip), (width, height))
    if not out_vid.isOpened():
        raise SystemExit(f"Could not open the video writer for {a.out_video}")

    csv_file = open(a.out_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["frame_idx", "det_idx", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2",
         "median_depth", "X", "Y", "Z", "dist_law_of_cosines", "dist_euclidean_3d"]
    )

    names = yolo.names
    frame_idx = processed = 0
    pbar = tqdm(total=total or None, desc="Frames")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pbar.update(1)
            if frame_idx % a.skip:
                frame_idx += 1
                continue

            res = yolo.predict(source=frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
            else:
                boxes, scores, classes = np.empty((0, 4)), np.empty(0), np.empty(0, int)

            depth = get_depth(depth_pipe, frame, width, height)
            out_frame = frame.copy()
            anchors = {}  # class_id -> camera coords, for the distance pair

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cls_id, conf = int(classes[i]), float(scores[i])
                z = bbox_median_depth(depth, x1, y1, x2, y2, a.window)
                u, v = (x1 + x2) / 2, (y1 + y2) / 2  # anchor = bbox midpoint
                cam = to_camera_coords(u, v, z, K) if z is not None else (None, None, None)
                if z is not None and cls_id in (a.drone_class, a.target_class):
                    anchors.setdefault(cls_id, cam)

                cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                txt = f"{names.get(cls_id, cls_id)} {conf:.2f} " + (f"z={z:.2f}m" if z is not None else "z=N/A")
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_frame, (int(x1), int(y1) - th - 6), (int(x1) + tw, int(y1)), (0, 255, 0), -1)
                cv2.putText(out_frame, txt, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                writer.writerow([frame_idx, i, cls_id, names.get(cls_id, ""), round(conf, 4),
                                 *[round(float(t), 2) for t in (x1, y1, x2, y2)],
                                 None if z is None else round(z, 4),
                                 *[None if c is None else round(float(c), 4) for c in cam], None, None])

            # --- distance between the drone and the target ---
            d_cam, t_cam = anchors.get(a.drone_class), anchors.get(a.target_class)
            if d_cam and t_cam:
                d_loc = distance_law_of_cosines(d_cam, t_cam)
                d_3d = distance_euclidean_3d(d_cam, t_cam)
                label = f"D(law-of-cos)={d_loc:.2f}m  D(3D)={d_3d:.2f}m"
                cv2.putText(out_frame, label, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                writer.writerow([frame_idx, "pair", a.drone_class, "drone->target", None,
                                 None, None, None, None, None, None, None, None,
                                 round(d_loc, 4), round(d_3d, 4)])

            # depth thumbnail, top-left
            vis = depth.astype(np.float32)
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            tw_ = max(1, min(320, width // 3))
            th_ = max(1, int(tw_ * height / width))
            out_frame[0:th_, 0:tw_] = cv2.addWeighted(out_frame[0:th_, 0:tw_], 0.7, cv2.resize(vis, (tw_, th_)), 0.3, 0)

            out_vid.write(out_frame)
            processed += 1
            frame_idx += 1
            if a.show:
                cv2.imshow("YOLOv8 + Depth Anything V2", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pbar.close()
        cap.release()
        out_vid.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"Done. {processed} frames -> {a.out_video}, {a.out_csv}")


if __name__ == "__main__":
    main()
