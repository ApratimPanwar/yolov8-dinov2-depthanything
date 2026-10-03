"""Does adding DINOv2 actually help?

Trains the combined model and plain YOLOv8n in exactly the same way, from the same
starting weights, and compares how well they detect and how often they fire on
nothing. The page claims DINOv2 cuts down wrong detections but never checks it.

    python ab_experiment.py --data construction-ppe.yaml --epochs 50 --batch 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from train_dino import load_pretrained

HERE = Path(__file__).resolve().parent


def background_fp(metrics):
    """Count boxes the model drew where there was nothing, and objects it missed.

    The last row and column of the confusion matrix are the background, so the final
    column counts detections on empty space.
    """
    try:
        m = metrics.confusion_matrix.matrix
        return float(m[:-1, -1].sum()), float(m[-1, :-1].sum())  # FP, FN(missed)
    except Exception:
        return float("nan"), float("nan")


def run_arm(name, build, a):
    model = build()
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        seed=a.seed, deterministic=True, amp=False, workers=a.workers,
        project=str(HERE / "outputs"), name=name, exist_ok=True, verbose=False, plots=True,
    )
    metrics = model.val(data=a.data, project=str(HERE / "outputs"), name=name + "_val", exist_ok=True)
    fp, fn = background_fp(metrics)
    r = metrics.results_dict
    return {
        "arm": name,
        "mAP50": round(float(r["metrics/mAP50(B)"]), 4),
        "mAP50-95": round(float(r["metrics/mAP50-95(B)"]), 4),
        "precision": round(float(r["metrics/precision(B)"]), 4),
        "recall": round(float(r["metrics/recall(B)"]), 4),
        "background_FP": fp,
        "missed_FN": fn,
        "params": sum(p.numel() for p in model.model.parameters()),
        # NOTE: counted after val(), where Ultralytics has already frozen everything, so
        # this is not the training-time figure. Stock trains all 3.16 M; the fused model
        # trains 3.39 M of 25.44 M (22.06 M frozen DINOv2).
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="construction-ppe.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weights", default="yolov8n.pt")
    a = ap.parse_args()

    results = []

    # Arm A: stock YOLOv8n, fine-tuned from the same pretrained checkpoint.
    results.append(run_arm("ab_stock", lambda: YOLO(a.weights), a))

    # Arm B: fused YOLOv8n + frozen DINOv2, warm-started from the same checkpoint
    # via the index-remapping loader.
    def build_fused():
        m = YOLO(str(HERE / "yolov8n-dino.yaml"))
        load_pretrained(m, a.weights)
        return m

    results.append(run_arm("ab_fused_dino", build_fused, a))

    out = HERE / "outputs" / "ab_results.json"
    out.write_text(json.dumps({"config": vars(a) | {"device": str(a.device)}, "results": results}, indent=2))

    print("\n" + "=" * 78)
    print(f"A/B on {a.data} — {a.epochs} epochs, batch {a.batch}, seed {a.seed}")
    print("=" * 78)
    hdr = f"{'arm':<16}{'mAP50':>9}{'mAP50-95':>10}{'prec':>8}{'recall':>8}{'bg FP':>9}{'missed':>9}"
    print(hdr); print("-" * 78)
    for r in results:
        print(f"{r['arm']:<16}{r['mAP50']:>9.4f}{r['mAP50-95']:>10.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['background_FP']:>9.0f}{r['missed_FN']:>9.0f}")
    s, f = results[0], results[1]
    print("-" * 78)
    print(f"delta (fused - stock): mAP50 {f['mAP50']-s['mAP50']:+.4f}   "
          f"mAP50-95 {f['mAP50-95']-s['mAP50-95']:+.4f}   "
          f"background FP {f['background_FP']-s['background_FP']:+.0f}")
    print(f"total params: stock {s['params']:,}  fused {f['params']:,} "
          f"({f['params']/s['params']:.1f}x)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
