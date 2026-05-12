import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

FUSION_DIR     = "/workspace/fusion_results"
DATA_INFO      = "/workspace/CMeRT/data/data_info.json"
RISK_THRESHOLD = 0.5

with open(DATA_INFO) as f:
    test_sessions = json.load(f)["UCFCrime"]["test_session_set"]

video_pred = []
video_true = []
rows = []

for session in test_sessions:
    risk_path = os.path.join(FUSION_DIR, session + "_risk.npy")
    if not os.path.exists(risk_path):
        print(f"[SKIP] {session}")
        continue

    true_binary = 0 if os.path.basename(session).startswith("Normal") else 1
    risk        = np.load(risk_path).astype(np.float32)
    N           = len(risk)
    threat_frames = int((risk >= RISK_THRESHOLD).sum())
    pred_binary   = 1 if threat_frames > (N - threat_frames) else 0

    video_pred.append(pred_binary)
    video_true.append(true_binary)
    rows.append({
        "video":         session,
        "true":          true_binary,
        "pred":          pred_binary,
        "mean_risk":     round(float(risk.mean()), 4),
        "max_risk":      round(float(risk.max()),  4),
        "threat_frames": threat_frames,
        "total_frames":  N,
        "correct":       true_binary == pred_binary,
    })

video_pred = np.array(video_pred)
video_true = np.array(video_true)

pd.DataFrame(rows).to_csv("video_level_results.csv", index=False)

cm        = confusion_matrix(video_true, video_pred)
cm_norm   = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
precision = precision_score(video_true, video_pred, zero_division=0)
recall    = recall_score(   video_true, video_pred, zero_division=0)
f1        = f1_score(       video_true, video_pred, zero_division=0)
accuracy  = accuracy_score( video_true, video_pred)

print(f"Processed : {len(rows)} videos")
print(f"Normal    : {(video_true == 0).sum()}")
print(f"Threat    : {(video_true == 1).sum()}")
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Threat"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Threat"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Binary Confusion Matrix (Fusion — Video Level)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                color="white" if cm_norm[i, j] > 0.5 else "black")
plt.tight_layout()
plt.savefig("video_cm_binary.png", dpi=300)
plt.close()
print("Saved: video_cm_binary.png")
