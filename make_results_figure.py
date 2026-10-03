"""Draw the results chart comparing the two models.

    python make_results_figure.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "assignment1_yolo_dino" / "outputs"


def rows(path):
    with open(path, newline="") as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ab = json.loads((OUT / "ab_results.json").read_text())["results"]
    stock, fused = ab[0], ab[1]
    s_hist, f_hist = rows(OUT / "ab_stock" / "results.csv"), rows(OUT / "ab_fused_dino" / "results.csv")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Does adding DINOv2 help? Plain YOLOv8n vs YOLOv8n + DINOv2, construction site PPE\n"
                 "1132 training and 143 test images, 11 classes, 50 epochs, same start and same seed",
                 fontsize=13, fontweight="bold")

    ep = lambda h: [int(r["epoch"]) for r in h]
    g = lambda h, k: [float(r[k]) for r in h]

    a = ax[0, 0]
    a.plot(ep(s_hist), g(s_hist, "metrics/mAP50(B)"), label="stock YOLOv8n")
    a.plot(ep(f_hist), g(f_hist, "metrics/mAP50(B)"), label="YOLOv8n + DINOv2")
    a.set_xlabel("Training epoch"); a.set_ylabel("mAP@0.5 on test images (0 to 1)")
    a.set_title("(a) How well each model detects, as training goes on"); a.legend(); a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(ep(s_hist), g(s_hist, "val/cls_loss"), label="stock YOLOv8n")
    a.plot(ep(f_hist), g(f_hist, "val/cls_loss"), label="YOLOv8n + DINOv2")
    a.set_xlabel("Training epoch"); a.set_ylabel("Error on test images (lower is better)")
    a.set_title("(b) Classification error"); a.legend(); a.grid(alpha=.3)

    a = ax[1, 0]
    labels = ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]
    sv = [stock["mAP50"], stock["mAP50-95"], stock["precision"], stock["recall"]]
    fv = [fused["mAP50"], fused["mAP50-95"], fused["precision"], fused["recall"]]
    x = range(len(labels)); w = .36
    a.bar([i - w / 2 for i in x], sv, w, label="stock", color="#888")
    a.bar([i + w / 2 for i in x], fv, w, label="+DINOv2", color="#2980b9")
    for i, (u, v) in enumerate(zip(sv, fv)):
        a.text(i - w / 2, u + .012, f"{u:.3f}", ha="center", fontsize=8)
        a.text(i + w / 2, v + .012, f"{v:.3f}", ha="center", fontsize=8)
    a.set_xticks(list(x)); a.set_xticklabels(labels); a.set_ylim(0, .75)
    a.set_ylabel("Score (0 to 1)"); a.set_title("(c) Final scores. The gaps are too small to mean anything")
    a.legend(); a.grid(axis="y", alpha=.3)

    a = ax[1, 1]
    names = ["boxes drawn on\nnothing", "objects\nmissed"]
    sv = [stock["background_FP"], stock["missed_FN"]]
    fv = [fused["background_FP"], fused["missed_FN"]]
    x = range(len(names))
    a.bar([i - w / 2 for i in x], sv, w, label="stock", color="#888")
    a.bar([i + w / 2 for i in x], fv, w, label="+DINOv2", color="#c0392b")
    for i, (u, v) in enumerate(zip(sv, fv)):
        a.text(i - w / 2, u + 4, f"{u:.0f}", ha="center", fontsize=9, fontweight="bold")
        a.text(i + w / 2, v + 4, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks(list(x)); a.set_xticklabels(names)
    a.set_ylabel("Count on test images (lower is better)")
    a.set_title("(d) The page says DINOv2 cuts wrong detections.\nIt cut 9 out of 228, and the model got 8.4x bigger")
    a.legend(); a.grid(axis="y", alpha=.3)

    fig.tight_layout()
    out = HERE / "results_assignment1_dino.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out, f"({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
