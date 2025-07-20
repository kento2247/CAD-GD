import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold, threshold_box
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.criterion_box import SetCriterionBox, SetCriterionFSC147
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def threshold(
        outputs,
        captions: str,
        tokenizer,
        threshold1 = 0.25,
        threshold2 = 0.20): 

    bs = outputs["pred_logits"].shape[0]

    ret = []
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]  
        prediction_boxes = outputs["pred_boxes"].cpu()[b]  

        tokenized = tokenizer(captions[b])
        input_ids = tokenized['input_ids']
        end_idx = np.where(np.array(input_ids)==1012)[0][-1]
        
        # find mask index where all the valid tokens are above the threshold
        threshold1 = threshold1
        threshold2 = threshold2
        # for global context
        mask1 = prediction_logits[:, 0].gt(threshold1)
        mask2 = prediction_logits[:, 1:end_idx].gt(threshold2).all(dim=1) 
        mask = mask1 & mask2

        logits = prediction_logits[mask]  
        boxes = prediction_boxes[mask]  


        ret.append((boxes, logits.max(dim=1)[0]))

    return ret

def eval(split, model, loaders, args):
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
    for images, patches, targets, boxes, texts, file_name in tqdm(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        density_maps = targets['density_map'].squeeze(1).to(device)
        boxes = boxes.to(torch.float32).to(device)

        # points = targets['points'].to(device)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=texts)

        pred_density_map = outputs['density'].squeeze(1)

        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])
        
        for i, p in enumerate(gt_num_density):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2
        
        density_maps.shape
        
        total_num += density_maps.shape[0]

        # localization target and loss
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        h, w = images.shape[-2:]
        shape = [h, w]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(targets, emb_size, shape)

        counter_for_image += 1
        results = threshold_box(outputs, args.threshold)
        
        pred_num_judge = len(results[0][1])
        if pred_num_judge >= args.pred_num_judge:
            h, w = images.shape[-2:]
            r_images = []
            resize_transform = T.Resize(800)
            r_images.append(resize_transform(TF.crop(images[0], 0, 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], 0, int(w/2), int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), int(w/2), int(h/2), int(w/2))))
            
            points_list = []
            logits_list = []
            density_pred_list = []
            for r_image in r_images:
                with torch.no_grad():
                    outputs = model(r_image.unsqueeze(0), captions=texts)
                    outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
                    result_temp = threshold_box(outputs, threshold=args.threshold)
                    points_list.append(result_temp[0][0])
                    logits_list.append(result_temp[0][1])
                    density_pred_list.append(torch.sum(outputs['density'].squeeze(1), dim=[1,2]) / args.scale)
            points_1 = torch.cat(points_list, dim=0)
            logits_1 = torch.cat(logits_list, dim=0)
            new_pred_num_judge = len(logits_1)
            new_results = [[points_1, logits_1]]
            
            new_pred_density = sum(density_pred_list)

            if new_pred_density < 4 * pred_num:
                results = new_results
        
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
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN

            counter += 1
    
    eval_mae_regression = round(val_mae.item()/total_num,2)
    eval_rmse_regression = round(val_rmse.item()/total_num,2)**0.5

    eval_mae = eval_mae / total_num
    eval_rmse = (eval_rmse / total_num) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression

def eval_reproduce(split, model, loaders, args):
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
    for images, patches, targets, boxes, texts, file_name in tqdm(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        # if total_num > 10:
        #     break
        density_maps = targets['density_map'].squeeze(1).to(device)
        boxes = boxes.to(torch.float32).to(device)

        # points = targets['points'].to(device)
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images, captions=texts)
        ## 计算loss
        
        total_num += density_maps.shape[0]

        # localization target and loss
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        h, w = images.shape[-2:]
        shape = [h, w]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(targets, emb_size, shape)

        counter_for_image += 1
        results = threshold_box(outputs, args.threshold)
        
        pred_num_judge = len(results[0][1])
        # if abs(pred_num_judge - len(targets[0]["points"])) > 100:
        #     print(pred_num_judge, len(targets[0]["points"]), file_name)
        if pred_num_judge >= args.pred_num_judge:
            # print(pred_num_judge, len(targets[0]["points"]), pred_num[0])
            h, w = images.shape[-2:]
            r_images = []
            resize_transform = T.Resize(800)
            # resized_img = resize_transform(img)
            r_images.append(resize_transform(TF.crop(images[0], 0, 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], 0, int(w/2), int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), int(w/2), int(h/2), int(w/2))))
            
            points_list = []
            logits_list = []
            for r_image in r_images:
                with torch.no_grad():
                    outputs = model(r_image.unsqueeze(0), captions=texts)
                    outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
                    result_temp = threshold_box(outputs, threshold=args.threshold)
                    points_list.append(result_temp[0][0])
                    logits_list.append(result_temp[0][1])
            points_1 = torch.cat(points_list, dim=0)
            logits_1 = torch.cat(logits_list, dim=0)
            new_pred_num_judge = len(logits_1)
            new_results = [[points_1, logits_1]]

            results = new_results

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
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN

            counter += 1

    eval_mae = eval_mae / total_num
    eval_rmse = (eval_rmse / total_num) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1

def prepare_targets(anno_b, emb_size, shape):
    ## TODO: 生成对应的结果
    gt_points_b = [points/np.array(shape)[::-1] for points in anno_b['points']]
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 
    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 
    
    for i, l in enumerate(gt_logits):
        gt_logits[i][:,:3] = 1.0 
    
    caption_sizes = [torch.tensor(3)] * len(gt_logits)
    
    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size":caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

    return targets


def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
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
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1