import sys
sys.path.append('./src')
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.models import build_model


def do_perframe_det(cfg, model, device, logger):
    

    class_names = [
        "Normal",
        "Abuse",
        "Arrest",
        "Arson",
        "Assault",
        "Burglary",
        "Explosion",
        "Fighting",
        "RoadAccidents",
        "Robbery",
        "Shooting",
        "Shoplifting",
        "Stealing",
        "Vandalism"
    ]
    

    def extract_class_from_filename(filename):
        filename = str(filename)
        for idx, cname in enumerate(class_names):
            if cname == "Normal":
                if filename.startswith("Normal"):
                    return idx
            elif filename.startswith(cname):
                return idx
        return -1
    
    model.eval()
    
    dataset = build_dataset(
        cfg,
        phase='test'
    )
   

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )
   =
    print("\n================ DATASET ATTRIBUTES ================\n")
    print(dir(dataset))
    
    if hasattr(dataset, 'sessions'):
        video_name_list = dataset.sessions
    else:
        raise Exception(
            "Dataset does not contain sessions attribute."
        )
    
    video_predictions = {}
    
    with torch.no_grad():
        pbar = tqdm(
            data_loader,
            desc='Video Inference'
        )
        for batch_idx, batch in enumerate(pbar):
            
            inputs = [x.to(device) for x in batch[:-1]]
            
            scores, fut_scores = model(*inputs)
            
            score = scores[-1]
            score = score.softmax(dim=-1)
            score = score.cpu().numpy()   
            
            batch_size = score.shape[0]
            start_idx  = batch_idx * cfg.DATA_LOADER.BATCH_SIZE
            
            for bs in range(batch_size):
                session_idx = start_idx + bs
                if session_idx >= len(video_name_list):
                    break
                session = str(video_name_list[session_idx])
                if session not in video_predictions:
                    video_predictions[session] = []
               
                frame_preds = np.argmax(
                    score[bs],
                    axis=-1
                )
                
                video_predictions[session].extend(
                    frame_preds.flatten().tolist()
                )
    
    video_preds = []
    video_gts   = []
    print("\n================ VIDEO PREDICTIONS ================\n")
    for video_name, preds in video_predictions.items():
        
        pred_class = Counter(preds).most_common(1)[0][0]
        
        gt_class = extract_class_from_filename(video_name)
        if gt_class == -1:
            print(f"Could not determine GT for {video_name}")
            continue
        video_preds.append(pred_class)
        video_gts.append(gt_class)
        print(f"VIDEO : {video_name}")
        print(f"GT    : {class_names[gt_class]}")
        print(f"PRED  : {class_names[pred_class]}")
        print("------------------------------------------------")
    
    if len(video_preds) == 0:
        logger.info("No valid videos to evaluate.")
        return
    
    labels = list(range(len(class_names)))
    cm = confusion_matrix(video_gts, video_preds, labels=labels)
    
    cm_norm   = cm.astype(float)
    row_sums  = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm   = cm_norm / row_sums
    
    print("\n============= VIDEO LEVEL CONFUSION MATRIX =============\n")
    print(f"{'Class':<18}" + "".join([f"{c[:6]:>8}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<18}" + "".join([f"{x:>8}" for x in row]))
    
    total = np.sum(cm)
    print(f"\n{'Class':<18}{'Precision':>10}{'Recall':>10}{'F1':>8}{'Accuracy':>12}")
    rows = []
    for i in range(len(class_names)):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - (TP + FP + FN)
        pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        re = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
        ac = (TP + TN) / total if total > 0 else 0.0
        print(f"{class_names[i]:<18}{pr:>10.3f}{re:>10.3f}{f1:>8.3f}{ac:>12.3f}")
        rows.append([class_names[i], pr, re, f1, ac])
    pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Accuracy"]).to_csv(
        "video_level_per_class_metrics.csv", index=False
    )
    
    vbt    = (np.array(video_gts)   != 0).astype(int)
    vbp    = (np.array(video_preds) != 0).astype(int)
    cm_bin = confusion_matrix(vbt, vbp, labels=[0, 1])
    vtn, vfp, vfn, vtp = cm_bin.ravel()
    vprec = vtp / (vtp + vfp) if (vtp + vfp) > 0 else 0.0
    vrec  = vtp / (vtp + vfn) if (vtp + vfn) > 0 else 0.0
    vacc  = (vtp + vtn) / np.sum(cm_bin)
    vf1   = 2 * vprec * vrec / (vprec + vrec) if (vprec + vrec) > 0 else 0.0
    print(f"\n===== BINARY VIDEO-LEVEL (Threat vs Non-Threat) =====")
    print(f"Confusion Matrix:\n{cm_bin}")
    print(f"Precision: {vprec:.4f}  Recall: {vrec:.4f}  Accuracy: {vacc:.4f}  F1: {vf1:.4f}")
    
    np.save("video_level_cm.npy",      cm)
    np.save("video_level_cm_norm.npy", cm_norm)
    np.save("video_level_binary_cm.npy", cm_bin)
    

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        "video_level_confusion_matrix.csv"
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Video-Level Confusion Matrix (Normalised)")
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig("video_level_confusion_matrix.png", dpi=300)
    plt.close()
    

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.imshow(cm_bin, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Non-Threat", "Threat"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Non-Threat", "Threat"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title("Binary Video-Level Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_bin[i, j]),
                     ha="center", va="center", fontsize=14, color="white")
    plt.tight_layout()
    plt.savefig("video_level_binary_cm.png", dpi=300)
    plt.close()
    
    accuracy = np.mean(np.array(video_preds) == np.array(video_gts))
    print("\n======================================")
    print(f"VIDEO LEVEL ACCURACY : {accuracy:.4f}")
    print("======================================")
    logger.info(
        f"Video-Level Accuracy: {accuracy:.5f} | "
        f"Precision: {vprec:.4f}  Recall: {vrec:.4f}  F1: {vf1:.4f}"
    )

def infer(cfg):
    
    device       = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger       = setup_logger(cfg, phase='test')
    
    model = build_model(cfg, device)
   
    checkpointer.load(model)
    
    do_perframe_det(cfg, model, device, logger)

if __name__ == '__main__':
    cfg = load_cfg()
    infer(cfg)
