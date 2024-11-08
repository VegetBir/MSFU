# from medpy import metric
# import numpy as np
# import torch
# from scipy.spatial.distance import directed_hausdorff

# def calculate_indicators(predictions, labels):
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#     predictions = predictions.flatten()
#     labels = labels.flatten()
#     for i in range(len(predictions)):
#         if predictions[i] == 1 and labels[i] == 1:
#             tp += 1
#         elif predictions[i] == 1 and labels[i] == 0:
#             fp += 1
#         elif predictions[i] == 0 and labels[i] == 0:
#             tn += 1
#         elif predictions[i] == 0 and labels[i] == 1:
#             fn += 1
#
#     if (tp + fn == 0 or tp + fp == 0 or tp + fn + fp == 0 or tp == 0 or tp + fp + fn == 0 or tp + tn == 0 or tp + fp + fn + tn == 0):
#         com = 0
#         cor = 0
#         q = 0
#         f1 = 0
#         iou = 0
#         acc = 0
#     else:
#         com = tp / (tp + fn)
#         cor = tp / (tp + fp)
#         q = tp / (tp + fn + fp)
#         f1 = 2 * com * cor / (com + cor)
#         iou = tp / (tp + fp + fn)
#         acc = (tp + tn) / (tp + fp + fn + tn)
#     return com, cor, q, f1, iou, acc
#
# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#
#     pred_np = pred.cpu().detach().numpy()
#     gt_np = gt.cpu().detach().numpy()
#
#     if pred.sum() > 0 and gt.sum() > 0:
#         bce_diceloss = metric.binary.dc(pred_np, gt_np)
#         # hd95 = metric.binary.hd95(pred_np, gt_np)
#
#         com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
#         return bce_diceloss, com, cor, q, f1,iou,acc
#     elif pred.sum() > 0 and gt.sum() == 0:
#         com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
#         return 1, com, cor, q, f1,iou,acc
#     else:
#         com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
#         return 0, com, cor, q, f1,iou,acc
#
#
#
# # [0,~]
# def compute_hd95(pred, label):
#     pred = pred.squeeze().cpu().numpy()
#     label = label.squeeze().cpu().numpy()
#     distances = []
#     for i in range(pred.shape[0]):
#         d = directed_hausdorff(pred[i], label[i])[0]
#         distances.append(d)
#     distances = np.array(distances)
#     hd95 = np.percentile(distances, 95)
#     return hd95

import torch
import math
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# [0,1]
def compute_dice_score(pred, targs):
    pred = (pred>0).float()
    dice_socre = 2. * (pred*targs).sum() / (pred+targs).sum()
    return dice_socre

def compute_com(pred,label):
    TP = (pred * label).sum(dim=[2, 3])
    FN = ((1 - pred) * label).sum(dim=[2, 3])
    smooth = 1e-5
    com = TP / (TP+FN+smooth)
    com = torch.mean(com,dim=0).item()
    return com

def compute_cor(pred,label):
    TP = (pred * label).sum(dim=[2, 3])
    FP = (pred * (1 - label)).sum(dim=[2, 3])
    smooth = 1e-5
    cor = TP / (TP+FP+smooth)
    cor = torch.mean(cor,dim=0).item()
    return cor

def compute_f1(pred,label):
    com = compute_com(pred,label)
    cor = compute_cor(pred,label)
    f1 = (2.*com*cor) / (com + cor)
    return f1

# [0,~]
def compute_hd95(pred, label):
    pred = pred.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()
    distances = []
    for i in range(pred.shape[0]):
        d = directed_hausdorff(pred[i], label[i])[0]
        distances.append(d)
    distances = np.array(distances)
    hd95 = np.percentile(distances, 95)
    return hd95

# [0,1]
def compute_acc(pred,label):
    label = label.squeeze(1)  # shape [b, 256, 256]
    pred = pred.squeeze(1)  # shape [b, 256, 256]

    correct_pixels = torch.sum(label == pred).item()  # count number of correct pixels
    total_pixels = label.numel()  # count total number of pixels
    accuracy = correct_pixels / total_pixels  # compute accuracy
    return accuracy

# [0,1]
def compute_iou(pred, label):
    # 计算交集和并集
    intersection = torch.sum(pred * label)
    union = torch.sum((pred + label) > 0)

    # 计算IOU
    iou = intersection / union

    return iou


def calculate_metric_percase(img, label):
    diceScore = compute_dice_score(img, label)
    hd95 = compute_hd95(img, label)
    com = compute_com(img, label)
    cor = compute_cor(img, label)
    acc = compute_acc(img, label)
    iou = compute_iou(img, label)

    return  diceScore, hd95, com, cor, acc, iou