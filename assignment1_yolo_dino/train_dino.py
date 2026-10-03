"""Project 1: train the YOLOv8 + DINOv2 hybrid detector.

Run patch_ultralytics.py first. Example:
    python train_dino.py --data coco8.yaml --epochs 10 --imgsz 640 --batch 8
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def remap_key(k):
    """Map a stock YOLOv8 state_dict key onto this architecture's layer numbering.

    Adding a block at the start pushes the first half down by one, and the extra layer
    in the middle pushes the second half down by four. Without this, loading the normal
    weights only matches 7 of 355 values, so the model starts from scratch.
    """
    parts = k.split(".")
    if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
        return None
    i = int(parts[1])
    if i <= 9:  # backbone
        j = i + 1
    elif i <= 22:  # neck + Detect
        j = i + 4
    else:
        return None
    return ".".join(["model", str(j), *parts[2:]])


def load_pretrained(model, weights):
    """Transfer stock YOLOv8 weights into the fused model, by remapped name and shape."""
    src = YOLO(weights).model.state_dict()
    dst = model.model.state_dict()
    transferred = {}
    for k, v in src.items():
        j = remap_key(k)
        if j in dst and dst[j].shape == v.shape:
            transferred[j] = v
    dst.update(transferred)
    model.model.load_state_dict(dst)

    # Layer 13 is new, so it has no weights to copy and starts off random. That ruins
    # what the pre-trained weights were doing: mAP drops from 0.888 to 0.170. Setting it
    # to pass the YOLO half straight through, and ignore the DINOv2 half at first, fixes
    # it. Layer 12 joins them in the order [DINOv2, YOLO].
    sq = model.model.model[13].conv
    c_out = sq.weight.shape[0]
    with torch.no_grad():
        sq.weight.zero_()
        for i in range(c_out):
            sq.weight[i, c_out + i, 0, 0] = 1.0  # let the YOLO half through, ignore DINOv2

    # Ultralytics only passes a model you built yourself to the trainer if it thinks the
    # model came from a file. Built from a YAML it does not, so it rebuilds from scratch and
    # throws away everything loaded above. Pretending there is a checkpoint stops that.
    model.ckpt = {"model": model.model, "epoch": -1}

    n_dino = sum(1 for _ in model.model.model[11].parameters())
    print(f"Pretrained transfer: {len(transferred)}/{len(src)} tensors from {weights}; "
          f"layer 13 identity-initialised; layer 11 DINOv2 ({n_dino} tensors) left frozen")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(HERE / "yolov8n-dino.yaml"), help="scale is read from the filename (n/s/m/l/x)")
    ap.add_argument("--data", default="coco8.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640, help="must be a multiple of 32")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default="yolov8n_dino")
    ap.add_argument("--project", default=str(HERE / "outputs"), help="output folder")
    ap.add_argument("--pretrained", default="yolov8n.pt",
                    help="stock YOLOv8 weights to warm-start from; 'none' trains from scratch")
    a = ap.parse_args()

    if a.imgsz % 32:
        raise SystemExit(f"--imgsz must be a multiple of 32 (got {a.imgsz}); DINOv2 fuses at the stride-32 grid")

    model = YOLO(a.cfg)
    if a.pretrained and a.pretrained.lower() != "none":
        load_pretrained(model, a.pretrained)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        name=a.name,
        project=a.project,
        amp=False,  # the frozen DINOv2 trunk makes the AMP allclose check unreliable
    )
    m = model.val(project=a.project, name=a.name + "_val")
    print({k: round(float(v), 5) for k, v in m.results_dict.items()})
    print(f"Outputs written to {a.project}")


if __name__ == "__main__":
    main()
