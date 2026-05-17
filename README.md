# Threat Detection from Surveillance Camera

This project combines **CMeRT** and **YOLO-based TTC (Time-To-Collision)** estimation for surveillance threat detection.

- CMeRT is used for online action detection
- YOLO is used for object tracking and TTC estimation
- A fusion model combines both outputs for improved threat detection

GitHub Repository: https://github.com/Kristpher/Threat-Detection-from-surveillance-camera

Base CMeRT Repository: https://github.com/pangzhan27/CMeRT

---

# Features

- Frame-level action prediction
- Video-level threat classification
- Binary threat detection
- TTC-based motion analysis
- Fusion of semantic and motion features

---

# Dataset

- THUMOS14
- UCF Crime Dataset

---

# Project Structure

```text
project/
│
├── configs/
├── checkpoints/
├── data/
│
├── cmert_outputs/
│   └── per_video/
│
├── ttc/
│   ├── results_yolo/
│   └── results_yolo_subsampled/
│
├── main.py
├── main_test.py
├── yolo.py
├── fusion_model.py
├── fusion_model_results.py
├── fusion_results/
└── README.md
```

---

# Installation

Clone the repositories:

```bash
git clone https://github.com/Kristpher/Threat-Detection-from-surveillance-camera.git

git clone https://github.com/pangzhan27/CMeRT.git
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install ultralytics
```

---

# Train CMeRT on THUMOS14

Download pre-extracted THUMOS14 features from TeSTra.

Run:

```bash
python main.py \
--config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml
```

---

# Train on UCF Crime

Create the required:
- JSON files
- YAML config files

Then run:

```bash
python main.py \
--config_file configs/UCFCrime/ucfcrime.yaml
```

---

# Inference and Confusion Matrix

```bash
python main_test.py \
--test 1 \
--config_file configs/UCFCrime/ucfcrime.yaml \
MODEL.CHECKPOINT checkpoints/UCFCrime/epoch-9.pth
```

This generates:
- Frame predictions
- Video predictions
- Confusion matrices
- Per-video outputs

---

# Generate TTC Values

Run:

```bash
python yolo.py
```

Outputs are stored in:

```text
ttc/results_yolo/
```

---

# Subsample TTC Outputs

Used to align TTC values with CMeRT frame predictions.

```python
import numpy as np
import os

TTC_DIR = "ttc/results_yolo"
CMERT_DIR = "cmert_outputs/per_video"
OUT_DIR = "ttc/results_yolo_subsampled"

os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(CMERT_DIR):

    if not fname.endswith(".npy"):
        continue

    stem = fname.replace(".npy", "")

    ttc = np.load(os.path.join(TTC_DIR, stem + "_ttc.npy"))
    cmert = np.load(os.path.join(CMERT_DIR, stem + ".npy"))

    idx = np.linspace(
        0,
        len(ttc) - 1,
        len(cmert)
    ).astype(int)

    ttc_sub = ttc[idx]

    np.save(
        os.path.join(OUT_DIR, stem + "_ttc.npy"),
        ttc_sub
    )
```

---

# Train Fusion Model

```bash
python fusion_model.py --mode train \
    --cmert_dir  cmert_outputs/per_video \
    --ttc_dir    ttc/results_yolo_subsampled \
    --label_dir  data/UCF_TEMP_DATA/target_perframe \
    --ckpt       ./fusion_best_yolo.pth
```

---

# Fusion Inference

```bash
python fusion_model.py --mode infer \
    --cmert_dir  cmert_outputs/per_video \
    --ttc_dir    ttc/results_yolo_subsampled \
    --ckpt       ./fusion_best_yolo.pth \
    --out_dir    ./fusion_results
```

---

# Final Evaluation

```bash
python fusion_model_results.py
```

Generates:
- Binary confusion matrix
- Multiclass confusion matrix
- Accuracy
- Precision
- Recall
- F1-score

---

# Video-Level Aggregation

CMeRT predicts frame-level classes.

For video-level classification:
- Majority voting is used
- If any abnormal class appears, the Normal class is overridden

Binary mapping:
- Normal → Non-Threat
- Other classes → Threat

---

# Fusion Formula

```text
Risk = a × OAD + c × (1 / TTC)
```

Where:
- OAD = CMeRT prediction
- TTC = Time-To-Collision estimate

---

# Results

Example binary performance:

| Metric | Value |
|---|---|
| Accuracy | 0.7707 |
| Precision | 0.8640 |
| Recall | 0.8504 |
| F1-Score | 0.8571 |

---
