import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import argparse
import csv
import warnings
warnings.filterwarnings("ignore")

CMERT_DIR   = "CMeRT/cmert_outputs/per_video_ANT"   
TTC_DIR     = "CMeRT/ttc/results_proxy_v2"           
LABEL_DIR   = "CMeRT/data/UCF_TEMP_DATA/target_perframe"
SAVE_PATH   = "./fusion_best_ANT.pth"                
TTC_CLIP_MIN   = 0.1
TTC_CLIP_MAX   = 30.0
RISK_THRESHOLD = 0.5
EPOCHS         = 50
LR             = 1e-3
BATCH_SIZE     = 512

def onehot_to_binary(label_path):

    oh = np.load(label_path)

    if oh.ndim == 1:
        return (oh > 0).astype(np.float32)

    cls = np.argmax(oh, axis=1)

    return (cls != 0).astype(np.float32)

class FusionDataset(Dataset):
    def __init__(self, cmert_dir, ttc_dir, label_dir):
        self.samples = []
        cmert_files = sorted(os.listdir(cmert_dir))
        matched = 0
        skipped = 0
        for fname in cmert_files:
            if not fname.endswith(".npy"):
                continue
            stem = fname.replace(".npy", "")
            ttc_path   = os.path.join(ttc_dir,   stem + "_ttc.npy")
            label_path = os.path.join(label_dir, stem + ".npy")
            if not os.path.exists(ttc_path):
                skipped += 1
                continue
            if not os.path.exists(label_path):
                skipped += 1
                continue
            oad   = np.load(os.path.join(cmert_dir, fname)).astype(np.float32)
            ttc   = np.load(ttc_path).astype(np.float32)
            label = onehot_to_binary(label_path)
            N = min(len(oad), len(ttc), len(label))
            oad, ttc, label = oad[:N], ttc[:N], label[:N]
            ttc_clipped = np.clip(ttc, TTC_CLIP_MIN, TTC_CLIP_MAX)
            inv_ttc = 1.0 / ttc_clipped
            x = np.stack([oad, inv_ttc], axis=1)
            self.samples.append((x, label))
            matched += 1
        print(f"Dataset: {matched} videos matched, {skipped} skipped")
        if matched == 0:
            raise RuntimeError("No matched videos found. Check your directory paths.")
        self.x = np.concatenate([s[0] for s in self.samples], axis=0)
        self.y = np.concatenate([s[1] for s in self.samples], axis=0)
        n_threat = int(self.y.sum())
        n_normal = len(self.y) - n_threat
        print(f"Total frames : {len(self.y):,}")
        print(f"  Normal     : {n_normal:,}  ({100*n_normal/len(self.y):.1f}%)")
        print(f"  Threat     : {n_threat:,}  ({100*n_threat/len(self.y):.1f}%)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class RiskFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_oad = nn.Linear(1, 1, bias=False)   
        self.gate  = nn.Linear(2, 1, bias=False)   
        self.bias  = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.w_oad.weight, 0.5)
        nn.init.constant_(self.gate.weight,  0.25)

    def forward(self, x):
        oad   = x[:, 0:1]                     
        ttc   = x[:, 1:2]                          
        gate  = torch.sigmoid(self.gate(x))        
        logit = self.w_oad(oad) + gate * ttc + self.bias
        return torch.sigmoid(logit).squeeze(1)     

    def get_logits(self, x):
        oad   = x[:, 0:1]
        ttc   = x[:, 1:2]
        gate  = torch.sigmoid(self.gate(x))
        return (self.w_oad(oad) + gate * ttc + self.bias).squeeze(1)

    def get_coefficients(self):
        w_oad = self.w_oad.weight.data.cpu().numpy().flatten()[0]
        g     = self.gate.weight.data.cpu().numpy().flatten()
        b     = self.bias.data.cpu().numpy().item()
        return {
            "a (OAD weight)": w_oad,
            "gate_oad":       g[0],
            "gate_ttc":       g[1],
            "bias":           b,
        }

def train(cmert_dir, ttc_dir, label_dir, save_path=SAVE_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print("\nLoading dataset...")
    dataset = FusionDataset(cmert_dir, ttc_dir, label_dir)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = RiskFusionModel().to(device)
    n_threat = int(dataset.y.sum())
    n_normal = len(dataset.y) - n_threat
    pos_weight = torch.tensor([n_normal / max(n_threat, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_f1   = 0.0
    print(f"\n{'Epoch':>5} | {'TrLoss':>8} | {'VaLoss':>8} | {'Acc':>6} | {'F1':>6} | {'AUC':>6}")
    print("-" * 52)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model.get_logits(x_batch)     
            loss   = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(x_batch)
        tr_loss /= n_train
        model.eval()
        va_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model.get_logits(x_batch) 
                va_loss += criterion(logits, y_batch).item() * len(x_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.cpu().numpy())
        va_loss  /= n_val
        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds      = (all_probs >= RISK_THRESHOLD).astype(int)
        acc = (preds == all_labels).mean()
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float("nan")
        f1 = f1_score(all_labels, preds, zero_division=0)
        scheduler.step(va_loss)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            marker = "  * best"
        print(f"{epoch:>5} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {acc:>6.3f} | {f1:>6.3f} | {auc:>6.3f}{marker}")
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    coeffs = model.get_coefficients()
    print("\n── Learned Coefficients ──────────────────────")
    for k, v in coeffs.items():
        print(f"  {k:25s}: {v:+.4f}")
    print(f"\n  Risk = {coeffs['a (OAD weight)']:+.4f} x OAD  +  gate({coeffs['gate_oad']:+.4f}, {coeffs['gate_ttc']:+.4f}) x (1/TTC)  +  {coeffs['bias']:+.4f}")
    return model

def infer(cmert_dir, ttc_dir, ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskFusionModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    coeffs = model.get_coefficients()
    print("\n── Loaded Coefficients ───────────────────────")
    for k, v in coeffs.items():
        print(f"  {k:25s}: {v:+.4f}")
    results = []
    for fname in sorted(os.listdir(cmert_dir)):
        if not fname.endswith(".npy"):
            continue
        stem     = fname.replace(".npy", "")
        ttc_path = os.path.join(ttc_dir, stem + "_ttc.npy")
        if not os.path.exists(ttc_path):
            print(f"  [SKIP] {stem} — no TTC file")
            continue
        oad = np.load(os.path.join(cmert_dir, fname)).astype(np.float32)
        ttc = np.load(ttc_path).astype(np.float32)
        N = min(len(oad), len(ttc))
        oad, ttc = oad[:N], ttc[:N]
        ttc_clipped = np.clip(ttc, TTC_CLIP_MIN, TTC_CLIP_MAX)
        inv_ttc = 1.0 / ttc_clipped
        x = torch.tensor(np.stack([oad, inv_ttc], axis=1), dtype=torch.float32, device=device)
        with torch.no_grad():
            risk = model(x).cpu().numpy()
        np.save(os.path.join(out_dir, stem + "_risk.npy"), risk)
        alert        = risk.max() >= RISK_THRESHOLD
        alert_frames = int((risk >= RISK_THRESHOLD).sum())
        results.append({
            "video":        stem,
            "frames":       N,
            "max_risk":     round(float(risk.max()), 4),
            "mean_risk":    round(float(risk.mean()), 4),
            "alert_frames": alert_frames,
            "alert":        "YES" if alert else "NO",
        })
        status = "ALERT" if alert else "safe"
        print(f"  {status:5s}  {stem:45s}  max_risk={risk.max():.3f}")
    csv_path = os.path.join(out_dir, "risk_summary.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {csv_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk Fusion: CMeRT + TTC")
    parser.add_argument("--mode",       choices=["train", "infer"], required=True)
    parser.add_argument("--cmert_dir",  default=CMERT_DIR)
    parser.add_argument("--ttc_dir",    default=TTC_DIR)
    parser.add_argument("--label_dir",  default=LABEL_DIR)
    parser.add_argument("--ckpt",       default=SAVE_PATH)
    parser.add_argument("--out_dir",    default="./fusion_results_ANT")
    args = parser.parse_args()
    if args.mode == "train":
        train(args.cmert_dir, args.ttc_dir, args.label_dir, args.ckpt)
    else:
        infer(args.cmert_dir, args.ttc_dir, args.ckpt, args.out_dir)
