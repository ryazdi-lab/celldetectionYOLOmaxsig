# Cell Detection with YOLO + MaxSigLayerNormalized

Official implementation of **"End-to-End Contrast-Aware Patch-Statistic Enhancement Inside YOLO for Microscopy Cell Detection"**.

This repository provides a plug-and-play contrast-aware feature modulation layer (**MaxSigLayerNormalized**) integrated into YOLOv8 and YOLOv11 for peripheral blood cell detection in microscopy images.

---

## Overview

Accurate blood cell detection in microscopy images is challenging due to low contrast, staining variability, and local artifacts, especially for small or subtle classes such as red blood cells (RBCs) and platelets.

**MaxSigLayerNormalized** is a lightweight, trainable contrast modulation layer that operates directly on intermediate feature maps using normalized per-patch statistics:

- Combines **center**, **median**, and **mean** patch responses
- Uses **MAD-based robust normalization**
- Uses **sigmoid-bounded learnable parameters**
- Remains differentiable almost everywhere
- Adds only a small parameter/GFLOP overhead

The layer is integrated via a custom **MaxSigC2f** block into the backbone and neck of YOLOv8 and YOLOv11 without modifying the detection head.

---

## Highlights

- 🔌 Plug-and-play integration into YOLOv8 / YOLOv11
- 🧠 Learned contrast enhancement inside the detector (no external preprocessing)
- 📊 Evaluated on **BCCD** and **OI_PBC**, plus cross-dataset transfer **TXL-PBC → BCCD**
- ⚖️ Compared against classical contrast baselines: **CLAHE** and **percentile clipping**
- ⚡ Small overhead: e.g. **+0.86 GFLOPs** and **+0.08M parameters** for YOLOv11s

---

## Main Results (Selected)

| Model / Setting | Metric | Baseline | + MaxSigLayerNormalized |
|---|---|---|---|
| YOLOv11s / BCCD | mAP@0.5 | 0.943 | **0.949** |
| YOLOv11s / BCCD | F1 | 0.879 | **0.904** |
| YOLOv8s / BCCD | mAP@0.5 | 0.938 | **0.952** |
| YOLOv11s / OI_PBC | mAP@0.5:0.95 | 0.876 | **0.881** |

> Gains are **configuration-dependent** and were obtained with a single training seed. See the paper for full tables and limitations.

---

## Datasets

The project uses publicly available datasets:

- **BCCD** — 364 images, 4,888 instances (RBC / WBC / Platelets)
- **OI_PBC** — 16,891 images, 298,850 instances across 12 classes
- **TXL-PBC** — used for cross-dataset evaluation (train on TXL-PBC, test on BCCD)

---

## Citation

```bibtex
@article{yazdi2026cell,
  title   = {End-to-End Contrast-Aware Patch-Statistic Enhancement Inside YOLO for Microscopy Cell Detection},
  author  = {Yazdi, Reza and Yaghoubi, Mehdi and Karbasi, Soheila},
  journal = {Jorjani Biomedicine Journal},
  year    = {2026}
}
```
