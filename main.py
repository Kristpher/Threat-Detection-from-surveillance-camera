# version 0, context sample rate is same as long term. now chage it to short-term sample rate
'''
Divide future into short-horizon(anticipation) and long-horizon(future)
--config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml
--test 1 --config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml MODEL.CHECKPOINT checkpoints/THUMOS/cmert_long256_work4_kinetics_1x/epoch-9.pth MODEL.LSTR.INFERENCE_MODE batch
'''
import sys

sys.path.append('./src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch.utils.data as data
import os.path as osp
from bisect import bisect_right
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import math
import copy
import os.path as osp
import argparse
import json

from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.ema import build_ema
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.evaluation.postprocessing import postprocessing as default_pp
from rekognition_online_action_detection.criterions import build_criterion
from sklearn.metrics import average_precision_score
from rekognition_online_action_detection.utils.registry import Registry

from rekognition_online_action_detection.datasets import build_data_loader, build_dataset
# for models
from rekognition_online_action_detection.models import transformer as tr
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.evaluation import compute_result_new, compute_result


def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):

    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    s1_w = cfg.SOLVER.get('S1_W', 0.2)
    s2_w = cfg.SOLVER.get('S2_W', 1.0)
    f_w = cfg.SOLVER.get('F_W', 0.5)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        losses_dict = {}
        for l_name in ['tot', 'ant_cls', 'det_cls', 'fut_cls']:
            losses_dict[l_name] = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        pred_scores, ant_pred_scores, fut_pred_scores = [], [], []
        gt_targets, ant_gt_targets, fut_gt_targets  = [], [], []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    det_target, ant_target, fut_target = data[-1]

                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0][0]
                    scores, fut_scores = model(*[x.to(device) for x in data[:-1]])
                    tot_target = det_target.to(device)
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        tot_target = torch.cat((det_target, ant_target), dim=1).to(device)

                    for i, detant_score in enumerate(scores):
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score, ant_score = detant_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :], \
                                                   detant_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            ant_score = ant_score.reshape(-1, cfg.DATA.NUM_CLASSES)

                        detant_score = detant_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            detant_loss = s1_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss = s1_w * criterion['MCE'](det_score, det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss = s1_w * criterion['MCE'](ant_score, ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            detant_loss += s2_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss += s2_w * criterion['MCE'](det_score,
                                                              det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss += s2_w * criterion['MCE'](ant_score,
                                                              ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                    fut_loss = torch.FloatTensor([0]).to(scores[0].device)[0]
                    for i, fut_score in enumerate(fut_scores):
                        fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            fut_loss = f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            fut_loss += f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))

                    loss = detant_loss + fut_loss
                    losses_dict['tot'][phase] += loss.item() * batch_size
                    losses_dict['fut_cls'][phase] += fut_loss.item() * batch_size
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        losses_dict['ant_cls'][phase] += ant_loss.item() * batch_size
                        losses_dict['det_cls'][phase] += det_loss.item() * batch_size

                    if training:
                        optimizer.zero_grad()
                        if loss.item() != 0:
                            loss.backward()
                            optimizer.step()
                            ema.update()
                            scheduler.step()
                    else:
                        # Prepare for evaluation
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score = det_score.softmax(dim=1).cpu().tolist()
                            det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                            ant_score = ant_score.softmax(dim=1).cpu().tolist()
                            ant_target = ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            ant_pred_scores.extend(ant_score)
                            ant_gt_targets.extend(ant_target)
                        else:
                            det_score = detant_score.softmax(dim=1).cpu().tolist()
                            det_target = tot_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                        if len(fut_scores) > 0 :
                            fut_score = fut_score.softmax(dim=1).cpu().tolist()
                            fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            fut_pred_scores.extend(fut_score)
                            fut_gt_targets.extend(fut_target)
        end = time.time()

        
        log = []
        log.append('Epoch {:2}'.format(epoch))
        train_log = '[train loss]'
        for k, v in losses_dict.items():
            train_log += ' {}: {:.3f},'.format(k, v['train'] / len(data_loaders['train'].dataset))
        log.append(train_log)

        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](cfg, gt_targets, pred_scores, )
            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                ant_result = compute_result['perframe'](cfg, ant_gt_targets, ant_pred_scores, )
            if len(fut_pred_scores)> 0:
                fut_result = compute_result['perframe'](cfg, fut_gt_targets, fut_pred_scores, )
            test_log = '[test loss]'
            for k, v in losses_dict.items():
                test_log += ' {}: {:.3f},'.format(k, v['test'] / len(data_loaders['test'].dataset))
            log.append(test_log)
            log.append('[mAP] det: {:.3f}， ant: {:.3f}, fut: {:.3f} '.format(det_result['mean_AP'],
                               ant_result['mean_AP'] if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0 else -1,
                                                    fut_result['mean_AP'] if len(fut_pred_scores) >0 else -1))

        log.append('running time: {:.2f} sec'.format(end - start, ))
        logger.info(' | '.join(log))

        
        if epoch % cfg.SOLVER.SAVE_EVERY == 0 and epoch >= 8:
            checkpointer.save(epoch, model, optimizer)

        if not training:
            ema.restore()

        
        data_loaders['train'].dataset.shuffle()


def main(cfg):
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='train')
    logger = setup_logger(cfg, phase='train')

    
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    
    model = build_model(cfg, device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    logger.info('Number of parameters: {}. Model Size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()),
                                                                         param_size / 1024 ** 2))

    
    criterion = build_criterion(cfg, device)

    
    optimizer = build_optimizer(cfg, model)

    
    ema = build_ema(model, 0.999)

    
    checkpointer.load(model, optimizer)

    
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    do_perframe_det_train(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        ema,
        device,
        checkpointer,
        logger,
    )







def do_perframe_det(cfg, model, device, logger):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from tqdm import tqdm
    import torch

    class_names = [
        "Normal","Abuse","Arrest","Arson","Assault","Burglary",
        "Explosion","Fighting","RoadAccidents","Robbery",
        "Shooting","Shoplifting","Stealing","Vandalism"
    ]

    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    det_pred, det_gt = [], []
    ant_pred, ant_gt = [], []
    fut_pred, fut_gt = [], []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Inference"):

            inputs = [x.to(device) for x in data[:-1]]

            det_target, ant_target, fut_target = data[-1]
            det_target = det_target.to(device)
            ant_target = ant_target.to(device)
            fut_target = fut_target.to(device)

            scores, fut_scores = model(*inputs)
            last_scores = scores[-1]

            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                wm = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

                det_score = last_scores[:, :wm, :].softmax(dim=-1)
                ant_score = last_scores[:, wm:, :].softmax(dim=-1)

                det_pred.append(det_score.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())
                det_gt.append(det_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())

                ant_pred.append(ant_score.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())
                ant_gt.append(ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())
            else:
                det_score = last_scores.softmax(dim=-1)
                det_pred.append(det_score.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())
                det_gt.append(det_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())

            if len(fut_scores) > 0:
                fut_score = fut_scores[-1].softmax(dim=-1)
                fut_pred.append(fut_score.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())
                fut_gt.append(fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().numpy())

    
    def compute_metrics(name, pred_list, gt_list):
        if len(pred_list) == 0:
            return

        pred = np.concatenate(pred_list, axis=0)
        gt = np.concatenate(gt_list, axis=0)
        import os; os.makedirs("cmert_outputs", exist_ok=True)
        threat_prob = (1.0 - pred[:, 0]).astype(np.float32)
        np.save(f"cmert_outputs/{name}_scores.npy", threat_prob)
        np.save(f"cmert_outputs/{name}_gt.npy", gt.astype(np.float32))

        y_true = np.argmax(gt, axis=1)
        y_pred = np.argmax(pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        np.save(f"{name}_cm.npy", cm)
        np.save(f"{name}_cm_norm.npy", cm_norm)

       
        plt.figure(figsize=(12, 10))
        plt.imshow(cm_norm)
        plt.title(f"{name} Confusion Matrix")
        plt.colorbar()

        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png", dpi=300)
        plt.close()

        
        total = np.sum(cm)
        rows = []

        print(f"\n===== {name} PER-CLASS METRICS =====")
        print(f"{'Class':<18}{'Precision':>10}{'Recall':>10}{'Accuracy':>12}")

        for i in range(len(class_names)):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP
            TN = total - (TP + FP + FN)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / total if total > 0 else 0

            print(f"{class_names[i]:<18}{precision:>10.3f}{recall:>10.3f}{accuracy:>12.3f}")
            rows.append([class_names[i], precision, recall, accuracy])

        pd.DataFrame(rows, columns=["Class","Precision","Recall","Accuracy"])\
            .to_csv(f"{name}_per_class_metrics.csv", index=False)

        
        y_true_bin = (y_true != 0).astype(int)
        y_pred_bin = (y_pred != 0).astype(int)

        cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

        TN, FP, FN, TP = cm_bin.ravel()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / np.sum(cm_bin)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n===== {name} BINARY (THREAT vs NON-THREAT) =====")
        print("Confusion Matrix:")
        print(cm_bin)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        
        np.save(f"{name}_binary_cm.npy", cm_bin)

        pd.DataFrame([{
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1": f1
        }]).to_csv(f"{name}_binary_metrics.csv", index=False)

        
        plt.figure()
        plt.imshow(cm_bin)
        plt.title(f"{name} Binary CM")
        plt.colorbar()
        plt.xticks([0,1], ["Non-Threat","Threat"])
        plt.yticks([0,1], ["Non-Threat","Threat"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{name}_binary_cm.png", dpi=300)
        plt.close()

        
        result = compute_result['perframe'](cfg, gt, pred)
        logger.info(f"{name} mAP: {result['mean_AP']:.5f}")

    
    compute_metrics("DET", det_pred, det_gt)
    compute_metrics("ANT", ant_pred, ant_gt)
    compute_metrics("FUT", fut_pred, fut_gt)



def infer(cfg):
    
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')

    
    model = build_model(cfg, device)

    
    checkpointer.load(model)

    do_perframe_det(
        cfg,
        model,
        device,
        logger,
    )


if __name__ == '__main__':
    cfg = load_cfg()
    if not cfg.TEST:
        main(cfg)
    else:
        infer(cfg)

