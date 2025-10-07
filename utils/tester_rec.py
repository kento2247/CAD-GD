import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from GroundingDINO.util.base_api import (
    load_model,
    threshold,
    threshold_box,
)
import os
import numpy as np
from datetime import datetime
import math

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.criterion_box import SetCriterionBox, SetCriterionFSC147
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from tqdm import tqdm
import torchvision.transforms.functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion_localization = SetCriterion()
criterion_counting = L2Loss()


def threshold_dynamic(outputs, pred_num):
    bs = outputs["pred_logits"].shape[0]
    ret = []
    b = 0
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]
        prediction_boxes = outputs["pred_boxes"].cpu()[b]

        logits = prediction_logits[:, 0]
        topk = pred_num[b]
        values, indices = torch.topk(logits.squeeze(), topk)
        ret.append((prediction_boxes[indices], logits[indices]))

    return ret


def threshold_dynamic_context(outputs, captions: str, tokenizer, pred_num):
    bs = outputs["pred_logits"].shape[0]
    ret = []
    b = 0
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]
        prediction_boxes = outputs["pred_boxes"].cpu()[b]

        tokenized = tokenizer(captions[b])
        input_ids = tokenized["input_ids"]
        end_idx = np.where(np.array(input_ids) == 1012)[0][-1]

        # find mask index where all the valid tokens are above the threshold
        prediction_logits_context = torch.mean(prediction_logits[:, 1:end_idx], dim=1)

        # logits = prediction_logits[:,0]
        logits = prediction_logits_context
        topk = pred_num[b]
        values, indices = torch.topk(logits.squeeze(), topk)
        ret.append((prediction_boxes[indices], logits[indices]))

    return ret


def eval(split, model, loaders, annotations, args):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    val_mae = 0
    val_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0

    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    total_num = 0

    for images, captions, shapes, img_caps, density_maps in tqdm(
        loader
    ):  # tensor, list, list, list
        mask_bi = [
            i for i, img_cap_list in enumerate(img_caps) for _ in img_cap_list
        ]  # index for each img,cap pair in the batch
        anno_b = [
            annotations[img_cap]
            for img_cap_list in img_caps
            for img_cap in img_cap_list
        ]
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [
            shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list
        ]
        images = torch.stack(
            [
                images[i]
                for i, caption_list in enumerate(captions)
                for _ in caption_list
            ],
            dim=0,
        )
        captions = [
            caption for caption_list in captions for caption in caption_list
        ]  # flatten list of list
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=captions, image_name=img_caps[0][0])
        ## density output
        pred_density_map = outputs["density"].squeeze(1)
        density_maps = density_maps.to(pred_density_map.device)
        ### mae
        total_num += density_maps.shape[0]
        counter_for_image += 1

        ## localization output
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(model, anno_b, captions, shapes, emb_size)
        # pred_num
        pred_num = torch.sum(pred_density_map, dim=[1, 2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim=[1, 2])
        for i, p in enumerate(targets):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err**2

        pred_num_round = [round(i.item()) for i in pred_num]

        # results = threshold_dynamic(outputs, pred_num_round)
        results = threshold_dynamic_context(
            outputs, captions, model.tokenizer, pred_num_round
        )
        # results = threshold(outputs, captions, model.tokenizer, text_threshold=0, threshold1=args.threshold1, threshold2=args.threshold2)
        for b in range(len(results)):
            boxes, logits = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err**2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(
                boxes, targets[b]["points"]
            )
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN

            counter += 1

            # print(f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae/counter)}, RMSE: {int((eval_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')

    eval_mae_regression = round(val_mae.item() / total_num, 2)
    eval_rmse_regression = round(val_rmse.item() / total_num, 2) ** 0.5

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = (
        2 * eval_precision * eval_recall / (eval_precision + eval_recall)
        if eval_precision + eval_recall != 0
        else 0.0
    )

    return (
        eval_mae,
        eval_rmse,
        eval_tp,
        eval_fp,
        eval_fn,
        eval_precision,
        eval_recall,
        eval_f1,
        eval_mae_regression,
        eval_rmse_regression,
    )


def prepare_targets(model, anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno["points"]) == 0:
            anno["points"] = [[0, 0]]
    gt_points_b = [
        np.array(anno["points"]) / np.array(shape)[::-1]
        for anno, shape in zip(anno_b, shapes)
    ]  # (h,w) -> (w,h)
    gt_points = [
        torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b
    ]

    gt_logits = [
        torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points
    ]

    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [
        torch.where(input_ids == 1012)[0][-1] for input_ids in tokenized["input_ids"]
    ]
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:, :end_idx] = 1.0

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [
        {
            "points": img_gt_points.to(device),
            "labels": img_gt_logits.to(device),
            "caption_size": caption_size,
        }
        for img_gt_points, img_gt_logits, caption_size in zip(
            gt_points, gt_logits, caption_sizes
        )
    ]

    return targets


def distance_threshold_func(boxes):  # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2] * box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas) // 2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0

    return threshold


def calc_loc_metric(pred_boxes, gt_points):  # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0

    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric="euclidean")

    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1

    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = (
        2 * (Precision * Recall) / (Precision + Recall)
        if Precision + Recall != 0
        else 0.0
    )

    return TP, FP, FN, Precision, Recall, F1
