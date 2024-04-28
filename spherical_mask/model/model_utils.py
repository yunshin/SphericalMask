import numpy as np
import torch
import torch.nn.functional as F

import torch_scatter
from ..ops import ballquery_batchflat
from scipy.stats import norm
import time
import pdb

def calc_square_dist(a, b, norm=True):
    """
    Calculating square distance between a and b
    a: [bs, n, c]
    b: [bs, m, c]
    """
    n = a.shape[1]
    m = b.shape[1]
    a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
    b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
    a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
    b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
    a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
    b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

    coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

    if norm:
        dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
        # dist = torch.sqrt(dist)
    else:
        dist = a_square + b_square - 2 * coor
        # dist = torch.sqrt(dist)
    return dist


def nms_and_merge(proposals_pred, scores, classes, threshold):
    proposals_pred = proposals_pred.float()  # (nProposal, N), float, cuda
    intersection = torch.mm(proposals_pred, proposals_pred.t())  # (nProposal, nProposal), float, cuda
    proposals_pointnum = proposals_pred.sum(1)  # (nProposal), float, cuda
    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
    ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)

    ixs = torch.argsort(scores, descending=True)

    pick = []
    proposals = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = classes[i]

        iou = ious[i, ixs[1:]]
        other_cls = classes[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        idx_to_merge = ixs[remove_ixs]
        proposals_to_merge = proposals_pred[idx_to_merge, :]
        n_proposals_to_merge = len(remove_ixs)
        proposal_merged = torch.sum(proposals_to_merge, dim=0) >= n_proposals_to_merge * 0.5

        proposals.append(proposal_merged)

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]

    pick = torch.tensor(pick, dtype=torch.long, device=scores.device)
    proposals = torch.stack(proposals, dim=0).bool()
    return pick, proposals

def standard_nms_with_iou(categories, scores, boxes, ious, threshold=0.2):
    ixs = torch.argsort(scores, descending=True)
    # n_samples = len(ixs)
    
   
    
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = categories[i]
        
        iou = ious[i, ixs[1:]]
        other_cls = categories[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        # condition = (iou > threshold)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]
    get_idxs = torch.tensor(pick, dtype=torch.long, device=scores.device)

    return  categories[get_idxs], scores[get_idxs], boxes[get_idxs], get_idxs


def standard_nms(proposals_pred, categories, scores, boxes, threshold=0.2):
    ixs = torch.argsort(scores, descending=True)
    # n_samples = len(ixs)
   
    intersection = torch.einsum("nc,mc->nm", proposals_pred.type(scores.dtype), proposals_pred.type(scores.dtype))
    proposals_pointnum = proposals_pred.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = categories[i]

        iou = ious[i, ixs[1:]]
        other_cls = categories[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        # condition = (iou > threshold)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]
    get_idxs = torch.tensor(pick, dtype=torch.long, device=scores.device)

    return proposals_pred[get_idxs], categories[get_idxs], scores[get_idxs], boxes[get_idxs], get_idxs


def matrix_nms(proposals_pred, categories, scores, boxes, final_score_thresh=0.1, topk=-1):
    if len(categories) == 0:
        return proposals_pred, categories, scores, boxes, []

    ixs = torch.argsort(scores, descending=True)
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    proposals_pred_sorted = proposals_pred[ixs]
    scores_sorted = scores[ixs]
    boxes_sorted = boxes[ixs]

    # (nProposal, N), float, cuda
    
    intersection = torch.einsum(
        "nc,mc->nm", proposals_pred_sorted.type(scores.dtype), proposals_pred_sorted.type(scores.dtype)
    )
    
    proposals_pointnum = proposals_pred_sorted.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)
    #ious_manual = get_iou(proposals_pred_sorted, proposals_pred_sorted)
    #if torch.allclose(ious, ious_manual) == False:
    #    pdb.set_trace()
    # label_specific matrix.
    categories_x = categories_sorted[None, :].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = ious * label_matrix

    # matrix nms
    sigma = 2.0
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient

    if topk != -1:
        _, get_idxs = torch.topk(
            cate_scores_update, k=min(topk, cate_scores_update.shape[0]), largest=True, sorted=False
        )
    else:
        get_idxs = torch.nonzero(cate_scores_update >= final_score_thresh).view(-1)

    return (
        proposals_pred_sorted[get_idxs],
        categories_sorted[get_idxs],
        cate_scores_update[get_idxs],
        boxes_sorted[get_idxs],
        get_idxs,
    )

def matrix_nms_wo_box(proposals_pred, categories, scores,  final_score_thresh=0.1, topk=-1):
    if len(categories) == 0:
        return proposals_pred, categories, scores,  []

    ixs = torch.argsort(scores, descending=True)
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    proposals_pred_sorted = proposals_pred[ixs]
    scores_sorted = scores[ixs]
   

    # (nProposal, N), float, cuda
    
    intersection = torch.einsum(
        "nc,mc->nm", proposals_pred_sorted.type(scores.dtype), proposals_pred_sorted.type(scores.dtype)
    )
    
    proposals_pointnum = proposals_pred_sorted.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)
    #ious_manual = get_iou(proposals_pred_sorted, proposals_pred_sorted)
    #if torch.allclose(ious, ious_manual) == False:
    #    pdb.set_trace()
    # label_specific matrix.
    categories_x = categories_sorted[None, :].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = ious * label_matrix

    # matrix nms
    sigma = 2.0
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient

    if topk != -1:
        _, get_idxs = torch.topk(
            cate_scores_update, k=min(topk, cate_scores_update.shape[0]), largest=True, sorted=False
        )
    else:
        get_idxs = torch.nonzero(cate_scores_update >= final_score_thresh).view(-1)

    return (
        proposals_pred_sorted[get_idxs],
        categories_sorted[get_idxs],
        cate_scores_update[get_idxs],
        get_idxs
    )
def matrix_nms_with_ious(categories, scores, boxes, ious,final_score_thresh=0.1, topk=-1):
    if len(categories) == 0:
        return categories, scores, boxes
    
    ixs = torch.argsort(scores, descending=True)
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    
    scores_sorted = scores[ixs]
    boxes_sorted = boxes[ixs]

    # (nProposal, N), float, cuda
    
    
    #ious_manual = get_iou(proposals_pred_sorted, proposals_pred_sorted)
    #if torch.allclose(ious, ious_manual) == False:
    #    pdb.set_trace()
    # label_specific matrix.
    categories_x = categories_sorted[None, :].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = ious * label_matrix

    # matrix nms
    sigma = 2.0
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient
    
    if topk != -1:
        _, get_idxs = torch.topk(
            cate_scores_update, k=min(topk, cate_scores_update.shape[0]), largest=True, sorted=False
        )
    else:
        get_idxs = torch.nonzero(cate_scores_update >= final_score_thresh).view(-1)

    return (
    
        categories_sorted[get_idxs],
        cate_scores_update[get_idxs],
        boxes_sorted[get_idxs],
    )


def nms(proposals_pred, categories, scores, boxes, test_cfg):
   
    if test_cfg.type_nms == "matrix":
        return matrix_nms(proposals_pred, categories, scores, boxes, topk=test_cfg.topk)
    elif test_cfg.type_nms == "standard":
        return standard_nms(proposals_pred, categories, scores, boxes, threshold=test_cfg.nms_threshold)
    else:
        raise RuntimeError("Invalid nms type")


def compute_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(inputs, targets, weights, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # return loss.sum()

    loss = (loss * weights).sum()
    return loss


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


@torch.no_grad()
def iou_aabb(pt_offsets_vertices=None, pt_offset_vertices_labels=None, coords=None, box_preds=None, box_gt=None):
    if coords is not None:
        coords_min_pred = coords + pt_offsets_vertices[:, 0:3]  # N x 3
        coords_max_pred = coords + pt_offsets_vertices[:, 3:6]  # N x 3

        coords_min_gt = coords + pt_offset_vertices_labels[:, 0:3]  # N x 3
        coords_max_gt = coords + pt_offset_vertices_labels[:, 3:6]  # N x 3
    else:
        coords_min_pred = box_preds[:, 0:3]  # n_queries x 3
        coords_max_pred = box_preds[:, 3:6]  # n_queries x 3

        coords_min_gt = box_gt[:, 0:3]  # n_inst x 3
        coords_max_gt = box_gt[:, 3:6]  # n_inst x 3

    upper = torch.min(coords_max_pred, coords_max_gt)  # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt)  # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1)  # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)
    return iou

def giou_aabb_polar(pt_offsets_vertices, pt_offset_vertices_labels, coords=None):
    if coords is None:
        # coords = torch.zeros((pt_offsets_vertices.shape[0], 3), dtype=torch.float, device=pt_offsets_vertices.device)
        coords = 0.0
    
    pt_offset_vertices_labels = pt_offset_vertices_labels.view(pt_offset_vertices_labels.shape[0],
                                                               int(pt_offset_vertices_labels.shape[1]/3),
                                                                3 )
    pt_offsets_vertices = pt_offsets_vertices.view(pt_offsets_vertices.shape[0],
                                                               int(pt_offsets_vertices.shape[1]/3),
                                                                3 )

    coords_pred = coords + pt_offsets_vertices
    coords_gt = coords + pt_offset_vertices_labels  
       
    pdb.set_trace()
    upper = torch.max(coords_pred, coords_gt)  # Nx3
    lower = torch.min(coords_pred, coords_gt)  # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1)  # N

    gt_volumes = torch.prod(torch.clamp((coords_gt ), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)

    upper_bound = torch.max(coords_pred, coords_gt)
    lower_bound = torch.min(coords_pred, coords_gt)

    volumes_bound = torch.prod(torch.clamp((upper_bound - lower_bound), min=0.0), -1)  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou
def giou_aabb(pt_offsets_vertices, pt_offset_vertices_labels, coords=None):
    if coords is None:
        # coords = torch.zeros((pt_offsets_vertices.shape[0], 3), dtype=torch.float, device=pt_offsets_vertices.device)
        coords = 0.0
    
    coords_min_pred = coords + pt_offsets_vertices[:, 0:3]  # N x 3
    coords_max_pred = coords + pt_offsets_vertices[:, 3:6]  # N x 3

    coords_min_gt = coords + pt_offset_vertices_labels[:, 0:3]  # N x 3
    coords_max_gt = coords + pt_offset_vertices_labels[:, 3:6]  # N x 3
   
       

    upper = torch.min(coords_max_pred, coords_max_gt)  # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt)  # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1)  # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)

    upper_bound = torch.max(coords_max_pred, coords_max_gt)
    lower_bound = torch.min(coords_min_pred, coords_min_gt)

    volumes_bound = torch.prod(torch.clamp((upper_bound - lower_bound), min=0.0), -1)  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


def cal_iou(volumes, x1, y1, z1, x2, y2, z2, sort_indices, index):
    rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

    xx1 = torch.index_select(x1, dim=0, index=sort_indices)
    xx2 = torch.index_select(x2, dim=0, index=sort_indices)
    yy1 = torch.index_select(y1, dim=0, index=sort_indices)
    yy2 = torch.index_select(y2, dim=0, index=sort_indices)
    zz1 = torch.index_select(z1, dim=0, index=sort_indices)
    zz2 = torch.index_select(z2, dim=0, index=sort_indices)

    # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
    # pivot = centroid[[index]]

    xx1 = torch.max(xx1, x1[index])
    yy1 = torch.max(yy1, y1[index])
    zz1 = torch.max(zz1, z1[index])
    xx2 = torch.min(xx2, x2[index])
    yy2 = torch.min(yy2, y2[index])
    zz2 = torch.min(zz2, z2[index])

    l = torch.clamp(xx2 - xx1, min=0.0)
    w = torch.clamp(yy2 - yy1, min=0.0)
    h = torch.clamp(zz2 - zz1, min=0.0)

    inter = w * h * l

    union = (rem_volumes - inter) + volumes[index]

    IoU = inter / union

    return IoU


def cal_giou(volumes, x1, y1, z1, x2, y2, z2, sort_indices, index):
    rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

    xx1 = torch.index_select(x1, dim=0, index=sort_indices)
    xx2 = torch.index_select(x2, dim=0, index=sort_indices)
    yy1 = torch.index_select(y1, dim=0, index=sort_indices)
    yy2 = torch.index_select(y2, dim=0, index=sort_indices)
    zz1 = torch.index_select(z1, dim=0, index=sort_indices)
    zz2 = torch.index_select(z2, dim=0, index=sort_indices)

    # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
    # pivot = centroid[[index]]

    xx1 = torch.max(xx1, x1[index])
    yy1 = torch.max(yy1, y1[index])
    zz1 = torch.max(zz1, z1[index])
    xx2 = torch.min(xx2, x2[index])
    yy2 = torch.min(yy2, y2[index])
    zz2 = torch.min(zz2, z2[index])

    l = torch.clamp(xx2 - xx1, min=0.0)
    w = torch.clamp(yy2 - yy1, min=0.0)
    h = torch.clamp(zz2 - zz1, min=0.0)

    inter = w * h * l

    union = (rem_volumes - inter) + volumes[index]

    IoU = inter / union

    x_min_bound = torch.min(xx1, x1[index])
    y_min_bound = torch.min(yy1, y1[index])
    z_min_bound = torch.min(zz1, z1[index])
    x_max_bound = torch.max(xx2, x2[index])
    y_max_bound = torch.max(yy2, y2[index])
    z_max_bound = torch.max(zz2, z2[index])

    convex_area = (x_max_bound - x_min_bound) * (y_max_bound - y_min_bound) * (z_max_bound - z_min_bound)
    gIoU = IoU - (convex_area - union) / (convex_area + 1e-6)

    return IoU, gIoU
def get_mask_iou(batch_mask, pred, gt):

    ious_list = []
    for batch_idx in range(len(pred)):

        pred_batch = pred[batch_idx]
        gt_batch = gt[batch_mask == batch_idx]
        

        num_instance = int(gt_batch.max()) + 1
        ious_list_batch = []
        for gt_idx in range(num_instance):
            gt_mask = gt_batch == gt_idx
            max_iou = 0
            for pred_idx in range(len(pred_batch)):
                pred_mask = torch.zeros(len(gt_mask)).cuda()
                if len(pred_batch[pred_idx]) == 0:
                    continue
                pred_mask[pred_batch[pred_idx]] = 1
                pred_mask = pred_mask == 1
                
                intersection = gt_mask * pred_mask
                iou = intersection.sum() / (gt_mask.sum() + pred_mask.sum() - intersection.sum() + 1e-6)
                
                if iou.item() > max_iou:
                    max_iou = iou.item()
            ious_list_batch.append(max_iou)
        pdb.set_trace()
        ious_list.append(ious_list_batch)
    pdb.set_trace()
    return ious_list
                

def get_points_inside_s_contour_vectorize(rays, centered_points_in_sph,  num_ray_width, num_ray_height, device='cuda', return_indice=False):

        #centered_points
        if device == 'cuda':
            xy_angles = torch.arange(0,360, int(360/num_ray_width)).cuda()
            yz_angles = torch.arange(0,180, int(180/num_ray_height)).cuda()
        else:
            xy_angles = torch.arange(0,360, int(360/num_ray_width))
            yz_angles = torch.arange(0,180, int(180/num_ray_height))
    
        unit_h = 180/num_ray_height
        unit_w = 360/num_ray_width

        diff_yz = centered_points_in_sph[:,1][:,None].repeat(1,len(yz_angles)) - yz_angles[None,:]
        diff_yz[diff_yz < 0 ] = unit_h + 1
        diff_yz = diff_yz / unit_h
        yz_min_val, yz_angles_num = torch.min(diff_yz,1)    

        diff_xy = centered_points_in_sph[:,2][:,None].repeat(1,len(xy_angles)) - xy_angles[None,:]
        diff_xy[diff_xy < 0 ] = unit_w + 1
        diff_xy = diff_xy / unit_w
        xy_min_val, xy_angles_num = torch.min(diff_xy, 1)

        angles_num = yz_angles_num*len(xy_angles) + xy_angles_num
         
        if device == 'cuda':
            points_bag = torch.zeros(len(angles_num) ,num_ray_height*num_ray_width).cuda()
        else:
            points_bag = torch.zeros(len(angles_num) ,num_ray_height*num_ray_width)
        
        points_bag[torch.arange(len(angles_num)).long(), angles_num.long() ] = centered_points_in_sph[:,0].float()   


        points_bag = points_bag.T # num_ray_width*num_ray_height X num_points
        
        points_bag_mask = (points_bag <= rays[:,None]) * (points_bag > 0)
        points_bag_mask = torch.sum(points_bag_mask*1,0)

        tot_indice_all = torch.where(points_bag_mask>0)[0]
        #points_bag_mask
        if return_indice:
            return tot_indice_all, angles_num
        return tot_indice_all
def batch_giou_cross(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: M, 6
    # out: N, M
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]
    intersection = torch.prod(
        torch.clamp(
            (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)

    volumes_bound = torch.prod(
        torch.clamp(
            (torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


def batch_giou_corres(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: N, 6
    # out: N, M
    # boxes1 = boxes1[:, None, :]
    # boxes2 = boxes2[None, :, :]
    
    intersection = torch.prod(
        torch.clamp(
            (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N
    
    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)

    volumes_bound = torch.prod(
        torch.clamp(
            (torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou

def get_iou_einsum(masks_a, masks_b):
    intersection = torch.einsum(
        "nc,mc->nm", masks_a, masks_b
    )
    pdb.set_trace()
    proposals_pointnum = masks_b.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

def get_iou(masks_a, masks_b, cnt_fp=False):

    iou_res = torch.zeros(len(masks_a), len(masks_b)).float().cuda()

    if cnt_fp :
        metric_res = torch.zeros(len(masks_a), len(masks_b),4).float().cuda()
        #always assume masks_a are gt
    for mask_idx in range(len(masks_a)):

        pred_mask = masks_a[mask_idx]
        pred_mask = pred_mask == 1

        for mask_idx2 in range(len(masks_b)):

            pred_mask2 = masks_b[mask_idx2]
            pred_mask2 = pred_mask2 == 1

            overlap = (pred_mask * pred_mask2).sum()
            iou = overlap / ( pred_mask.sum() + pred_mask2.sum() - overlap)
            iou_res[mask_idx, mask_idx2] = iou

            if cnt_fp:
                fp = (pred_mask == 0) * (pred_mask2 == 1)
                fn = (pred_mask == 1) * (pred_mask2 == 0)

                metric_res[mask_idx, mask_idx2, 0] = fp.sum()
                metric_res[mask_idx, mask_idx2, 1] = fn.sum()
                metric_res[mask_idx, mask_idx2, 2] = pred_mask.sum()
                metric_res[mask_idx, mask_idx2, 3] = pred_mask2.sum()
    if cnt_fp:
        return iou_res, metric_res
    return iou_res

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
def smooth_label(rays, fill_zero=True):
    #rays : 25,1

    if fill_zero:
        
        mask = rays > 1e-6
        zero_mask = rays == 1e-6

        zero_loc = torch.where(zero_mask)[0]
        valid_loc = torch.where(mask)[0]
        closest_vals = []
        
        for idx in range(len(zero_loc)):
            loc = zero_loc[idx]
            
           
            next_big = torch.where(((valid_loc - loc) > 0)*(rays[valid_loc] > 0))[0]
            if len(next_big) > 0:
                next_big = valid_loc[next_big]
                next_big = next_big[torch.argmin(next_big - loc)]
            else:
                next_big = loc
                
            before_small = torch.where(((valid_loc - loc) < 0))[0]
            if len(before_small) > 0:
                before_small = valid_loc[before_small]
                before_small = before_small[torch.argmin(before_small - loc)]
            else:
                before_small = loc
            bigger_value, smaller_value = rays[next_big,0], rays[before_small,0]

            if bigger_value > 1e-6 and smaller_value > 1e-6:
              
                fill_val = (bigger_value + smaller_value) / 2.0
            elif bigger_value == 1e-6 and smaller_value > 1e-6:
                fill_val = smaller_value
            elif bigger_value > 1e-6 and smaller_value == 1e-6:
                fill_val = bigger_value
            else:
                pdb.set_trace()
            closest_vals.append(fill_val.item())
        
        closest_vals = torch.tensor(np.array(closest_vals)).float().cuda()
        rays[zero_loc,0] = closest_vals
    
    rays[:,0] = torch.tensor(smooth(rays[:,0].cpu().numpy(),0.5)).float().cuda()
    
    return rays
def manual_iou(pred_masks, instance_labels, v2p_map, p2v_map, cls_pred, cls_label, p2v=False):

    max_num = instance_labels.max().item()
    gt_masks = []

    iou_res = torch.zeros(len(pred_masks), max_num).float().cuda()
    cls_accs = []
    closest_label = []
    for pred_idx in range(len(pred_masks)):

        pred_mask = pred_masks[pred_idx]
        pred_mask = pred_mask == 1
        inst_cls = cls_pred[pred_idx]
        
        if p2v == False:
            pred_mask_form = torch.zeros(len(instance_labels)).float().cuda()
            pred_mask_v = torch.where(pred_mask)[0]
            pred_mask_form[p2v_map[pred_mask_v].long()] = 1
            pred_mask = pred_mask_form == 1
           
        for gt_idx in range(max_num):

            if pred_idx == 0:
                gt_mask = instance_labels == gt_idx
                if p2v:
                    gt_mask_form = torch.zeros(len(pred_mask)).float().cuda()
                    gt_mask_v = torch.where(gt_mask)[0]
                    
                    gt_mask_form[v2p_map[gt_mask_v].long()] = 1.0
                    gt_mask = gt_mask_form == 1
                gt_masks.append(gt_mask)
            else:
                gt_mask = gt_masks[gt_idx]
            
          
            
            overlap = (gt_mask * pred_mask).sum()
            iou = overlap / ( gt_mask.sum() + pred_mask.sum() - overlap)
            iou_res[pred_idx, gt_idx] = iou
        max_iou_idx = torch.argmax(iou_res[pred_idx])
        gt_cls = cls_label[max_iou_idx]
        if gt_cls == -100:
            
            gt_cls = torch.tensor(18).long().cuda()
        closest_label.append(gt_cls)
        if gt_cls == inst_cls:
            cls_accs.append(1)
        else:
            cls_accs.append(0)
    
    iou_gt2pred = iou_res.T
    max_gt2pred, _ = torch.max(iou_gt2pred, 1)
    cls_accs = np.array(cls_accs)
    cls_accs = cls_accs.sum() / len(cls_accs)
    return max_gt2pred, iou_gt2pred, torch.stack(closest_label), cls_accs
   
def analyze_pred_ray(pred, cls_pred, conf_pred, gt, cls_gt, center_refine=False, ray_refine=False, cls_refine=False):
    #pred : N X (rays_width*rays_height + 1)X3
    #gt : M X (rays_width*rays_height + 1)X3
    

    pred_centers = pred[:,-1]
    gt_centers = gt[:,-1]

    pred = pred[:,:-1]
    gt = gt[:,:-1]

    pred2gt_dist = torch.cdist( pred_centers, gt_centers)
    pred2gt_dist, pred2gt_indice = torch.min(pred2gt_dist,1)

    gt2pred_dist = torch.cdist( gt_centers, pred_centers)
    gt2pred_dist, gt2pred_indice = torch.min(gt2pred_dist,1)

    #valid_mask = pred2gt_dist < 1.0
    #valid_chk = torch.zeros(len(gt)).long().cuda()
    
    #pred2gt_dist = pred2gt_dist[valid_mask]
    #pred2gt_indice = pred2gt_indice[valid_mask]
    #valid_chk[pred2gt_indice] = 1
    pred_centers = pred_centers[gt2pred_indice]
    pred = pred[gt2pred_indice]
    cls_pred = cls_pred[gt2pred_indice]
    conf_pred = conf_pred[gt2pred_indice]
    if center_refine:
        
        pred_centers = gt_centers#pred_centers[gt2pred_indice]
        
        #pred[gt2pred_indice]
        #pred = pred[gt2pred_indice]
        
        ###
    if ray_refine:
        #pred = gt[pred2gt_indice]
        #pred_centers = pred_centers[gt2pred_indice]
        #pred = pred[gt2pred_indice]
        pred = gt
    if cls_refine:
        '''
        
        gt_to_insert = cls_gt[pred2gt_indice].clone()
        gt_to_insert[gt_to_insert == -100] = 18
        f_col = torch.arange(len(pred)).long().cuda()
        cls_refine_arr = torch.cat([f_col[:,None], gt_to_insert[pred2gt_indice][:,None]],1)
        cls_pred[:,:] = -3
        cls_pred[cls_refine_arr[:,0], cls_refine_arr[:,1]] = 5
        conf_pred[:] = 0.99
        '''
       
        gt_to_insert = cls_gt.clone()
        gt_to_insert[gt_to_insert == -100] = 18
        f_col = torch.arange(len(pred)).long().cuda()
        
        cls_pred = torch.zeros(cls_gt.shape[0], cls_pred.shape[1]).float().cuda()
        conf_pred = torch.zeros(gt_to_insert.shape[0]).float().cuda()
        
        conf_pred[:] = 0.99
       
        cls_refine_arr = torch.cat([f_col[:,None], gt_to_insert[:,None]],1)
        cls_pred[:,:] = -3
        cls_pred[cls_refine_arr[:,0], cls_refine_arr[:,1]] = 5
        
    pred = torch.cat([pred, pred_centers[:,None,:]],1)
    
    return pred, cls_pred, conf_pred
    
def cdf_sequential(data,mean_std, cls_labels, return_cdf=True):

    result = np.zeros(data.shape)
    for cls_idx in range(21):

        
        indice = cls_labels ==cls_idx
        data_cls = data[indice]
        if cls_idx >= len(mean_std):
            continue
        mean, std = mean_std[cls_idx,:,0], mean_std[cls_idx,:,1]
        if indice.sum() > 0:
            for mean_idx in range(len(mean)):

                
                    
                
                if return_cdf:
                    value = norm.sf(x=data_cls[:, mean_idx], loc=mean[mean_idx], scale=std[mean_idx])
                    inverse_value = norm.ppf(value  , loc=-mean[mean_idx], scale=std[mean_idx])
                    #if abs((-1*inverse_value)-data_cls[:, mean_idx]).max() > 0.0001 :
                    #    pdb.set_trace()
                    result[indice, mean_idx] = value
                else:
                   
                    inverse_value = norm.ppf(value  , loc=-mean[mean_idx], scale=std[mean_idx])
                    result[indice, mean_idx] = inverse_value
    return result
def get_ray_cls_from_ray(ray, cls_pred, range_anchors, return_label=False):
   

    anchors_ = range_anchors[cls_pred]
    rays_only = ray[:,:-3]
    
    
    #anchors_max = torch.amax(anchors_,2)
    #chk = anchors_max < rays_only
    #if chk.sum().item()>0:
    #    wrong_indice = torch.where(chk) 
    #    pdb.set_trace()

    rays_only = rays_only[:,:,None].repeat(1,1,anchors_.shape[2])
    diff = anchors_ - rays_only
    
    #diff[diff < 0] = 100000000
    closest_bigger_anchors = torch.argmin(torch.abs(diff),2)

    

    anchors_flatten = anchors_.reshape(-1, range_anchors.shape[2])
    closest_anchors_flatten = closest_bigger_anchors.reshape(-1)

    anchors_form = torch.cat([torch.arange(closest_anchors_flatten.shape[0]).to(closest_anchors_flatten.device)[:,None],closest_anchors_flatten[:,None]],1)

    rays_from_anchors = anchors_flatten[anchors_form[:,0], anchors_form[:,1]]
    rays_from_anchors = rays_from_anchors.reshape(ray.shape[0], ray.shape[1]-3)
    
    rays_from_anchors = torch.cat([rays_from_anchors, ray[:,-3:]],1)
    
    if return_label:

        return rays_from_anchors, closest_anchors_flatten
    return rays_from_anchors
    #pdb.set_trace()
def get_iou_corres(pred_masks, gt_masks):
    
   
    pred_masks = torch.clamp(pred_masks,0,1)
    gt_masks = torch.clamp(gt_masks,0,1)

    intersection = torch.einsum("nc,mc->nm", pred_masks, gt_masks)
    proposals_pointnum1 = pred_masks.sum(1)  # (nProposal), float, cuda
    proposals_pointnum2 = gt_masks.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum1[None, :] + proposals_pointnum2[:, None] - intersection)

    ious_gt2pred = torch.amax(ious,0)

    fp = (gt_masks==0)*(pred_masks==1)
    fn = (gt_masks==1)*(pred_masks==0)

    fp = fp.sum(1)
    fn = fn.sum(1)
    print('fp:{0}'.format(fp))
    print('fn:{0}'.format(fn))
    
    print(ious_gt2pred)
    pdb.set_trace()
def get_mask_from_polar_single_inside_mask(all_points, rays_and_center, ray_mask_pred, num_range, num_width, num_height, is_angular=False, pointwise_angular=None, sim_ths=0.8, gt_locs=None):
    #with multiple points B X N
    masks_angular = []
    masks = []
    ############
    #ray_dist_pred = -1* ray_dist_pred
    ############
    
    
    for pred_idx in range(len(rays_and_center)):
        
        points = all_points[pred_idx]
        
        pred = rays_and_center[pred_idx]
        pred = pred.reshape(int(len(pred)/3),3)
        
        pred_center = pred[-1]
        
        pred = pred[:-1]
        
        pred_centered = pred - pred_center[None,:]

        rays = cartesian_to_spherical(pred_centered)
        rays = rays[:,3]

        points_centered = points - pred_center[None,:]
        
        points_centered_in_sph = cartesian_to_spherical(points_centered)
        points_centered_in_sph[:,4:] = torch.rad2deg(points_centered_in_sph[:,4:])
        points_centered_in_sph[:,5] += 180

        ray_mask_pred_i = ray_mask_pred[pred_idx]
        ray_mask_pred_i = ray_mask_pred_i.reshape(num_height*num_width, num_range)
        #pos_mask = ray_dist_pred[pred_idx] >= 0
        #points_centered_in_sph[pos_mask,3] = 0.01
        #points_centered_in_sph[pos_mask==False,3] = 100

        #pos_locs = torch.where(pos_mask)[0]
        
        
        
        points_centered_in_sph[:,3] = torch.clamp(points_centered_in_sph[:,3], 0.01 ,100)
        rays = torch.clamp(rays, 0.011, 100)
        indice, angles_num = get_points_inside_s_contour_vectorize(rays, points_centered_in_sph[:,3:], num_width, num_height, return_indice=True)
        
        ray_mask_pred_i_form = ray_mask_pred_i
        #ray_mask_pred_i_form = ray_mask_pred_i.reshape(num_width*num_height, num_range)
        points_dist = points_centered_in_sph[:,3]
        
        indice = get_mask_indice_inside_ray(points_dist, rays, ray_mask_pred_i_form, angles_num, num_height*num_width, num_range)
       
        
        '''
        mask_1 = torch.zeros(len(pos_mask)).float().cuda()
        mask_2 = torch.zeros(len(pos_mask)).float().cuda()
        mask_1[pos_locs] = 1
        mask_2[indice] = 1
        mask_ = mask_1 * mask_2
        indice = torch.where(mask_==1)[0]
        '''
        if is_angular:
            
            points_inside = points[indice]
            if len(pointwise_angular.shape) == 3:
                angles_inside = pointwise_angular[pred_idx,indice]
            else:
                angles_inside = pointwise_angular[indice]
            
            angles_est = get_angular_gt(points_inside, pred_center)
            #pdb.set_trace()
            cos_sim = F.cosine_similarity(angles_inside, angles_est )
            sim_mask = cos_sim > sim_ths
           
            indice_with_angle = indice[sim_mask]

            mask_angle = torch.zeros(len(points)).float().cuda()
            mask_angle[:] = -100
            mask_angle[indice_with_angle] = 1
            #mask_angle[sim_mask] = 1
            '''
            ious_gt = []
            for idx_gt in range(len(gt_locs)):
                mask_gt = torch.zeros(len(points)).float().cuda()
                mask_gt[gt_locs[idx_gt]] = 1
                iou = get_iou([mask_gt],[mask_angle])
                ious_gt.append(iou[0,0].item())
            pdb.set_trace()
            '''
            masks_angular.append(mask_angle[None,:])

        mask_ray = torch.zeros(len(points)).float().cuda()
        mask_ray[:] = -100
        mask_ray[indice] = 1
        masks.append(mask_ray[None,:])
    
    if is_angular:
        return torch.cat(masks), torch.cat(masks_angular)
    else:
        return torch.cat(masks)    
    
def mask2ray(dc_coords_float, mask_all,  rays_width, rays_height, angles_ref, center_provided=False, centers_=None):
    
    true_locs = torch.where(mask_all)
    all_locs = dc_coords_float[true_locs[1]]
    if center_provided == False:
        centers = torch_scatter.scatter_mean(all_locs, true_locs[0], out=torch.zeros(mask_all.shape[0], 3).float().cuda(),dim=0)
        #NX3
    else:
        centers = centers_
    
    
    centered_coords = dc_coords_float[None,:] - centers[:,None,:]
    
    new_pts_vec = cartesian_to_spherical(centered_coords)
    new_pts_vec[:,:,4:] = torch.rad2deg(new_pts_vec[:,:,4:])
    new_pts_vec[:,:,5] += 180

    new_pts_vec = new_pts_vec[:,:,3:]
    fore_locs = torch.where(mask_all==0)
    new_pts_vec[fore_locs[0], fore_locs[1],0] = 0.0
    dist_anchors, angles_num = divide_with_angle_vectorize_3d_ray(new_pts_vec, rays_width, rays_height)
    
    dist_anchors = torch.clamp(dist_anchors,1e-2,10)
    rays_with_center = torch.cat([dist_anchors, centers],1)
    return rays_with_center

def try_vectorize(dc_coords_float, mask_all, rays, rays_width, rays_height):
    
    centers = rays[:,-3:]
    centered_coords = dc_coords_float[None,:] - centers[:,None,:]
    new_pts_vec = cartesian_to_spherical(centered_coords)

    new_pts_vec[:,:,4:] = torch.rad2deg(new_pts_vec[:,:,4:])
    new_pts_vec[:,:,5] += 180

    
    dist_anchors, angles_num, roi_locs = divide_with_angle_vectorize_3d(new_pts_vec[:,:,3:], rays_width, rays_height, rays)
    return dist_anchors, angles_num, roi_locs, new_pts_vec

def find_sector(dc_coords_float, mask_all, rays, rays_width, rays_height, point_migration=False):
    
    centers = rays[:,-3:]
    centered_coords = dc_coords_float[None,:] - centers[:,None,:]
    new_pts_vec = cartesian_to_spherical(centered_coords)

    if point_migration:
        new_pts_vec[:,:,3] = new_pts_vec[:,:,3] - mask_all 
    
    new_pts_vec[:,:,4:] = torch.rad2deg(new_pts_vec[:,:,4:])
    new_pts_vec[:,:,5] += 180

    
    dist_anchors, angles_num, roi_locs = find_sector_(new_pts_vec[:,:,3:], rays_width, rays_height, rays)
   
    return dist_anchors, angles_num, roi_locs, new_pts_vec

def get_points_mask_from_ray(dc_coords_float, mask_all, rays, rays_width, rays_height, with_pred=False, pred_mask=None, mask_form=False, func_=None):
    #mask_all : NX1

    

    all_masks = []
    if with_pred:
        all_pred_mask = []
        mask_form = True
        
    if mask_form:
        final_pred_mask = torch.zeros_like(mask_all)
        final_pred_mask[:] = -100
        valid_indice = []
    for mask_idx in range(len(mask_all)):
        
        center = dc_coords_float[mask_all[mask_idx] == 1].mean(0)
        
        if with_pred:
            valid_indice_idx = []
        if mask_form:
            
            center = rays[mask_idx,-3:]
        centered_points = dc_coords_float - center[None,:]
        new_pts = cartesian_to_spherical(centered_points)
        old_pts = spherical_to_cartesian(new_pts[:,3:])
        ray_dists = rays[mask_idx]
    
        new_pts[:,4:] = torch.rad2deg(new_pts[:,4:])
        new_pts[:,5] += 180
        
        dist_label_anchors, angles_num_each = divide_with_angle_vectorize(new_pts[:,3:], rays_width, rays_height, return_ref=True, cartesian_ref=dc_coords_float)
      
        mask_each_ray = []
        valid_indice_idx = []
        if with_pred:
            pred_mask_each_ray = []
        for ray_idx in range(rays_width*rays_height):

            ray_angle = ray_dists[ray_idx]
            
            angles_mask = angles_num_each == ray_idx
            coords_ = new_pts[:,3]#[angles_mask]
            coords_mask = coords_ <= ray_angle
          

            valid_mask = angles_mask * coords_mask
            
            points_inside = mask_all[mask_idx,valid_mask]
            
            if with_pred:
                valid_locs = torch.where(valid_mask)[0]
                valid_indice_idx.append(valid_locs)
               
                pred_points_inside = pred_mask[mask_idx,valid_mask]
                pred_mask_each_ray.append(pred_points_inside)

                if mask_form:
                    
                    #pred_values = func_(F.relu(pred_points_inside[:,:,None]))
                    pred_values = pred_points_inside.squeeze()
                    pred_values = pred_values.squeeze()

                    final_pred_mask[mask_idx,valid_mask] = pred_values
            mask_each_ray.append(points_inside)
        
        
        valid_indice.append(torch.cat(valid_indice_idx))
        all_masks.append(mask_each_ray)
        if with_pred:
            all_pred_mask.append(pred_mask_each_ray)
    
    if with_pred:
        return all_masks, all_pred_mask, final_pred_mask, valid_indice
    if mask_form:
      
        final_pred_mask = final_pred_mask > 0.0
        final_pred_mask = final_pred_mask * 1.0
        final_pred_mask[final_pred_mask==0] = -100
       

        return final_pred_mask
    return all_masks
def get_mask_from_polar_single2(all_points, rays_and_center, ray_dist_pred, num_width, num_height, is_angular=False, pointwise_angular=None, sim_ths=0.8, gt_locs=None):
    #with multiple points B X N
    masks_angular = []
    masks = []
    ############
    ray_dist_pred = -1* ray_dist_pred
    ############
    

    for pred_idx in range(len(rays_and_center)):
        
        points = all_points[pred_idx]
        
        pred = rays_and_center[pred_idx]
        pred = pred.reshape(int(len(pred)/3),3)
        
        pred_center = pred[-1]
        
        pred = pred[:-1]
        
        pred_centered = pred - pred_center[None,:]

        rays = cartesian_to_spherical(pred_centered)
        rays = rays[:,3]

        points_centered = points - pred_center[None,:]
        
        points_centered_in_sph = cartesian_to_spherical(points_centered)
        points_centered_in_sph[:,4:] = torch.rad2deg(points_centered_in_sph[:,4:])
        points_centered_in_sph[:,5] += 180

        
        
        #pos_mask = ray_dist_pred[pred_idx] >= 0
        #points_centered_in_sph[pos_mask,3] = 0.01
        #points_centered_in_sph[pos_mask==False,3] = 100

        #pos_locs = torch.where(pos_mask)[0]
        
        neg_points_mask = ray_dist_pred[pred_idx] > 0
        pos_points_mask = ray_dist_pred[pred_idx] <=0
        #points_centered_in_sph[:,3] += ray_dist_pred[pred_idx]*10
        points_centered_in_sph[pos_points_mask,3] += ray_dist_pred[pred_idx, pos_points_mask]*10
        points_centered_in_sph[neg_points_mask,3] += ray_dist_pred[pred_idx, neg_points_mask]*10

        points_centered_in_sph[:,3] = torch.clamp(points_centered_in_sph[:,3], 0.01 ,100)
        rays = torch.clamp(rays, 0.011, 100)
        indice, angles_num = get_points_inside_s_contour_vectorize(rays, points_centered_in_sph[:,3:], num_width, num_height, return_indice=True)
      
        '''
        mask_1 = torch.zeros(len(pos_mask)).float().cuda()
        mask_2 = torch.zeros(len(pos_mask)).float().cuda()
        mask_1[pos_locs] = 1
        mask_2[indice] = 1
        mask_ = mask_1 * mask_2
        indice = torch.where(mask_==1)[0]
        '''
        if is_angular:
            
            points_inside = points[indice]
            if len(pointwise_angular.shape) == 3:
                angles_inside = pointwise_angular[pred_idx,indice]
            else:
                angles_inside = pointwise_angular[indice]
            
            angles_est = get_angular_gt(points_inside, pred_center)
            #pdb.set_trace()
            cos_sim = F.cosine_similarity(angles_inside, angles_est )
            sim_mask = cos_sim > sim_ths
           
            indice_with_angle = indice[sim_mask]

            mask_angle = torch.zeros(len(points)).float().cuda()
            mask_angle[:] = -100
            mask_angle[indice_with_angle] = 1
            #mask_angle[sim_mask] = 1
            '''
            ious_gt = []
            for idx_gt in range(len(gt_locs)):
                mask_gt = torch.zeros(len(points)).float().cuda()
                mask_gt[gt_locs[idx_gt]] = 1
                iou = get_iou([mask_gt],[mask_angle])
                ious_gt.append(iou[0,0].item())
            pdb.set_trace()
            '''
            masks_angular.append(mask_angle[None,:])

        mask_ray = torch.zeros(len(points)).float().cuda()
        mask_ray[:] = -100
        mask_ray[indice] = 1
        masks.append(mask_ray[None,:])
    
    if is_angular:
        return torch.cat(masks), torch.cat(masks_angular)
    else:
        return torch.cat(masks)
def get_mask_from_polar_single(points, rays_and_center, num_width, num_height, is_angular=False, pointwise_angular=None, sim_ths=0.8, gt_locs=None):

    masks_angular = []
    masks = []
    for pred_idx in range(len(rays_and_center)):
        
        
        pred = rays_and_center[pred_idx]
        pred = pred.reshape(int(len(pred)/3),3)
        
        pred_center = pred[-1]
        pred = pred[:-1]
        
        pred_centered = pred - pred_center[None,:]

        rays = cartesian_to_spherical(pred_centered)
        rays = rays[:,3]

        points_centered = points - pred_center[None,:]
        
        points_centered_in_sph = cartesian_to_spherical(points_centered)
        points_centered_in_sph[:,4:] = torch.rad2deg(points_centered_in_sph[:,4:])
        points_centered_in_sph[:,5] += 180

        
        indice = get_points_inside_s_contour_vectorize(rays, points_centered_in_sph[:,3:], num_width, num_height)

        if is_angular:
            
            points_inside = points[indice]
            if len(pointwise_angular.shape) == 3:
                angles_inside = pointwise_angular[pred_idx,indice]
            else:
                angles_inside = pointwise_angular[indice]
            
            angles_est = get_angular_gt(points_inside, pred_center)
            #pdb.set_trace()
            cos_sim = F.cosine_similarity(angles_inside, angles_est )
            sim_mask = cos_sim > sim_ths
           
            indice_with_angle = indice[sim_mask]

            mask_angle = torch.zeros(len(points)).float().cuda()
            mask_angle[:] = -100
            mask_angle[indice_with_angle] = 1
            #mask_angle[sim_mask] = 1
            '''
            ious_gt = []
            for idx_gt in range(len(gt_locs)):
                mask_gt = torch.zeros(len(points)).float().cuda()
                mask_gt[gt_locs[idx_gt]] = 1
                iou = get_iou([mask_gt],[mask_angle])
                ious_gt.append(iou[0,0].item())
            pdb.set_trace()
            '''
            masks_angular.append(mask_angle[None,:])

        mask_ray = torch.zeros(len(points)).float().cuda()
        mask_ray[:] = -100
        mask_ray[indice] = 1
        masks.append(mask_ray[None,:])
    
    if is_angular:
        return torch.cat(masks), torch.cat(masks_angular)
    else:
        return torch.cat(masks)
def get_mask_from_polar(points, rays_and_center, num_width, num_height, is_angular=False, pointwise_angular=None ):
    #points : NX4 , 0 - batch idx
    #rays_and_center: B X M X 48
    masks = []
    if is_angular:
        masks_angular = []
    for batch_idx in range(len(rays_and_center)):

        batch_points = points[points[:,0] == batch_idx][:,1:]
        if is_angular:
            batch_angular = pointwise_angular[points[:,0] == batch_idx]
            masks_angular_batch = []
        masks_batch = []

        for idx in range(len(rays_and_center[batch_idx])):
            
            pred = rays_and_center[batch_idx, idx]
            pred = pred.reshape(int(len(pred)/3),3)
            
            pred_center = pred[-1]
            pred = pred[:-1]
            
            pred_centered = pred - pred_center[None,:]

            rays = cartesian_to_spherical(pred_centered)
            rays = rays[:,3]

            points_centered = batch_points - pred_center[None,:]
            
            points_centered_in_sph = cartesian_to_spherical(points_centered)
            points_centered_in_sph[:,4:] = torch.rad2deg(points_centered_in_sph[:,4:])
            points_centered_in_sph[:,5] += 180

            
            indice = get_points_inside_s_contour_vectorize(rays, points_centered_in_sph[:,3:], num_width, num_height)

            if is_angular:
                
                points_inside = batch_points[indice]
                angles_inside = batch_angular[indice]
                angles_est = get_angular_gt(points_inside, pred_center)

                cos_sim = F.cosine_similarity(angles_inside, angles_est)
                sim_mask = cos_sim > 0.8
                indice_with_angle = indice[sim_mask]
                masks_angular_batch.append(indice_with_angle.cpu())
            masks_batch.append(indice.cpu())
        if is_angular:
            masks_angular.append(masks_angular_batch)    
        masks.append(masks_batch)
    if is_angular:
        return masks, masks_angular
    return masks
def batch_giou_cross_polar(boxes1, boxes2):
    # boxes1: N, rays_width * rays_width +1 , 3
    # boxes2: M, rays_width * rays_width +1 , 3
    # out: N, M
    # 
    # 
    # Getting IoU with polar mask's idea
    
    if len(boxes1.shape) == 2:
        boxes1 = boxes1.reshape(boxes1.shape[0], int(boxes1.shape[1]/3), 3)
        boxes2 = boxes2.reshape(boxes2.shape[0], int(boxes2.shape[1]/3), 3)

    boxes1_center = boxes1[:,-1]
    boxes2_center = boxes2[:,-1]

    boxes1 = boxes1[:,:-1]
    boxes2 = boxes2[:,:-1]

    #boxes1_dist = torch.sqrt(torch.sum(boxes1**2,2))
    #boxes2_dist = torch.sqrt(torch.sum(boxes2**2,2))

    boxes1_form = boxes1[:, None]
    boxes2_form = boxes2[None, :]

    
    

    boxes2_cen_from_cen1 = torch.sqrt(torch.sum((boxes1_center[:,None] - boxes2_center[None,:])**2,2))
    


    boxes1_from_cen1 = torch.sqrt(torch.sum( (boxes1_form - boxes1_center[:,None,None,:])**2,3))#256X1X15
    boxes2_from_cen1 = torch.sqrt(torch.sum( (boxes2_form - boxes1_center[:,None,None,:])**2,3))#256X11X15
    boxes2_from_cen1 += boxes2_cen_from_cen1[...,None]

    #boxes2_from_cen2 = torch.sqrt(torch.sum( (boxes2_form - boxes2_center[:,None,None,:])**2,3))
    boxes1_from_cen1 = boxes1_from_cen1.repeat(1,len(boxes2),1)

    
    merged_arrays = torch.cat([boxes1_from_cen1[...,None], boxes2_from_cen1[...,None]],3)

    iou_numerator = torch.sum(torch.amin(merged_arrays,3),2)
    iou_denominator = torch.sum(torch.amax(merged_arrays,3),2)
    ious = iou_numerator / (iou_denominator+1e-3)

    
    if False:#Deprecated method
        ious = torch.zeros(len(boxes2_dist),len(boxes1_dist)).to(boxes1.device)
        for idx in range(len(boxes2_dist)):


            
            iou_numerator = torch.sum(torch.min(boxes1_dist,boxes2_dist[idx][None,:].repeat(len(boxes1_dist),1)),1) 
            iou_denominator = torch.sum(torch.max(boxes1_dist,boxes2_dist[idx][None,:].repeat(len(boxes1_dist),1)),1)
            iou = iou_numerator / (iou_denominator + 1e-6)
            ious[idx] = iou
    #giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)
    
    #ious = ious.T
    return ious, ious

#def get_normalized_ray(rays, norm_params):

#def get_orig_ray_from_norm_ray(rays, norm_params):
#def get_anchors()



def batch_giou_cross_polar_exp1(boxes1, boxes2, angles_ref):
    # boxes1: N, rays_width * rays_width +1 , 3
    # boxes2: M, rays_width * rays_width +1 , 3
    # out: N, M
    # 
    # 
    # Getting IoU with polar mask's idea
    
    
    #boxes1 = boxes1.reshape(boxes1.shape[0], int(boxes1.shape[1]/3), 3)
    #boxes2 = boxes2.reshape(boxes2.shape[0], int(boxes2.shape[1]/3), 3)

    boxes1_center = boxes1[:,-3:]
    boxes2_center = boxes2[:,-3:]

    boxes1 = boxes1[:,:-3]
    boxes2 = boxes2[:,:-3]
    
    ###
    angles_ref_box1 = angles_ref[None,:,:]
    angles_ref_box1 = angles_ref_box1.repeat(len(boxes1),1,1)
    boxes1 = get_locs_from_ray(boxes1, boxes1_center, angles_ref_box1)
    ###

    angles_ref_box2 = angles_ref[None,:,:]
    angles_ref_box2 = angles_ref_box2.repeat(len(boxes2),1,1)
    
    boxes2 = get_locs_from_ray(boxes2, boxes2_center, angles_ref_box2)

    #boxes1_dist = torch.sqrt(torch.sum(boxes1**2,2))
    #boxes2_dist = torch.sqrt(torch.sum(boxes2**2,2))
    
    boxes1_form = boxes1[:, None]
    boxes2_form = boxes2[None, :]

    
    

    boxes2_cen_from_cen1 = torch.sqrt(torch.sum((boxes1_center[:,None] - boxes2_center[None,:])**2,2))
    


    boxes1_from_cen1 = torch.sqrt(torch.sum( (boxes1_form - boxes1_center[:,None,None,:])**2,3))#256X1X15
    boxes2_from_cen1 = torch.sqrt(torch.sum( (boxes2_form - boxes1_center[:,None,None,:])**2,3))#256X11X15
    boxes2_from_cen1 += boxes2_cen_from_cen1[...,None]

    #boxes2_from_cen2 = torch.sqrt(torch.sum( (boxes2_form - boxes2_center[:,None,None,:])**2,3))
    boxes1_from_cen1 = boxes1_from_cen1.repeat(1,len(boxes2),1)

    
    merged_arrays = torch.cat([boxes1_from_cen1[...,None], boxes2_from_cen1[...,None]],3)

    iou_numerator = torch.sum(torch.amin(merged_arrays,3),2)
    iou_denominator = torch.sum(torch.amax(merged_arrays,3),2)
    ious = iou_numerator / (iou_denominator+1e-3)

    
    if False:#Deprecated method
        ious = torch.zeros(len(boxes2_dist),len(boxes1_dist)).to(boxes1.device)
        for idx in range(len(boxes2_dist)):


            
            iou_numerator = torch.sum(torch.min(boxes1_dist,boxes2_dist[idx][None,:].repeat(len(boxes1_dist),1)),1) 
            iou_denominator = torch.sum(torch.max(boxes1_dist,boxes2_dist[idx][None,:].repeat(len(boxes1_dist),1)),1)
            iou = iou_numerator / (iou_denominator + 1e-6)
            ious[idx] = iou
    #giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)
    
    #ious = ious.T
    return ious, ious

def get_locs_from_ray(rays, centers, angles_ref):
    
    sph_coords = torch.cat([rays[:,:,None], angles_ref.to(rays.device) ],2)
    sph_coords[:,:,2] -= 180
    sph_coords[:,:,1:] = torch.deg2rad(sph_coords[:,:,1:])
    sph_points = spherical_to_cartesian(sph_coords)
    
    sph_points += centers[:,None,:]

    return sph_points

def batch_giou_corres_polar_exp1(boxes1, boxes2, angles_ref):
    # boxes1: N, rays_width * rays_height + 1(center)
    # boxes2: N, rays_width * rays_height + 1(center)
    # out: N, M
    # boxes1 = boxes1[:, None, :]
    # boxes2 = boxes2[None, :, :]
    # Getting IoU with polar mask's idea

    
    boxes1_center = boxes1[:,-3:]
    boxes2_center = boxes2[:,-3:]

    boxes1 = boxes1[:,:-3]
    boxes2 = boxes2[:,:-3]
   
    angles_ref_box1 = angles_ref[None,:,:]
    angles_ref_box1 = angles_ref_box1.repeat(len(boxes1),1,1)
    boxes1 = get_locs_from_ray(boxes1, boxes1_center, angles_ref_box1)
    
    angles_ref_box2 = angles_ref[None,:,:]
    angles_ref_box2 = angles_ref_box2.repeat(len(boxes2),1,1)
    boxes2 = get_locs_from_ray(boxes2, boxes2_center, angles_ref_box2)
    
    boxes2_cen_from_cen1 = torch.sqrt(torch.sum((boxes1_center - boxes2_center)**2,1))
    
    
    boxes1_from_cen1 = torch.sqrt(torch.sum( (boxes1-boxes1_center[:,None,:])**2,2))
    #if torch.isnan(boxes1_from_cen1).any().item() or torch.isinf(boxes1_from_cen1).any().item():
    #    pdb.set_trace() 
    
    boxes1_from_cen1 = torch.clamp(boxes1_from_cen1,1e-6, 1000)##### 0 causes NaN
    #print('dist max : {0} min : {1}'.format(torch.amax(boxes1_from_cen1).item(), torch.amin(boxes1_from_cen1).item()))
    
    boxes2_from_cen1 = torch.sqrt(torch.sum( (boxes2-boxes1_center[:,None,:])**2,2))
    boxes2_from_cen1_new = boxes2_from_cen1 + boxes2_cen_from_cen1[:,None]

    iou_numerator_new2 = torch.sum(torch.min(boxes1_from_cen1,boxes2_from_cen1_new),1) 
    iou_denominator_new2 = torch.sum(torch.max(boxes1_from_cen1,boxes2_from_cen1_new),1)
    iou_new = iou_numerator_new2 / (iou_denominator_new2 + 1e-6)
    
    
    #giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou_new, iou_new
def batch_giou_corres_polar(boxes1, boxes2):
    # boxes1: N, rays_width * rays_height + 1(center)
    # boxes2: N, rays_width * rays_height + 1(center)
    # out: N, M
    # boxes1 = boxes1[:, None, :]
    # boxes2 = boxes2[None, :, :]
    # Getting IoU with polar mask's idea

    
    boxes1_center = boxes1[:,-1]
    boxes2_center = boxes2[:,-1]

    boxes1 = boxes1[:,:-1]
    boxes2 = boxes2[:,:-1]

    
    boxes2_cen_from_cen1 = torch.sqrt(torch.sum((boxes1_center - boxes2_center)**2,1))
    boxes1_from_cen1 = torch.sqrt(torch.sum( (boxes1-boxes1_center[:,None,:])**2,2))
    boxes2_from_cen1 = torch.sqrt(torch.sum( (boxes2-boxes1_center[:,None,:])**2,2))
    boxes2_from_cen1_new = boxes2_from_cen1 + boxes2_cen_from_cen1[:,None]



    if False:#Deprecated method
        boxes1_dist = torch.sqrt(torch.sum(boxes1**2,2))
        boxes2_dist = torch.sqrt(torch.sum(boxes2**2,2))

        iou_numerator = torch.sum(torch.min(boxes1_dist,boxes2_dist),1) 
        iou_denominator = torch.sum(torch.max(boxes1_dist,boxes2_dist),1)
        iou = iou_numerator / (iou_denominator + 1e-6)


        iou_numerator_new = torch.sum(torch.min(boxes1_from_cen1,boxes2_from_cen1),1) 
        iou_denominator_new = torch.sum(torch.max(boxes1_from_cen1,boxes2_from_cen1),1)
        iou_new = iou_numerator_new / (iou_denominator_new + 1e-6)

    iou_numerator_new2 = torch.sum(torch.min(boxes1_from_cen1,boxes2_from_cen1_new),1) 
    iou_denominator_new2 = torch.sum(torch.max(boxes1_from_cen1,boxes2_from_cen1_new),1)
    iou_new = iou_numerator_new2 / (iou_denominator_new2 + 1e-6)
    
    
    #giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou_new, iou_new
def superpoint_align(spp, proposals_pred):
    if len(proposals_pred.shape) == 2:
        n_inst, n_points = proposals_pred.shape[:2]
        spp_unique, spp_ids = torch.unique(spp, return_inverse=True)

        mean_spp_inst = torch_scatter.scatter(
            proposals_pred.float(), spp_ids.expand(n_inst, n_points), dim=-1, reduce="mean"
        )  # n_inst, n_spp
        spp_mask = mean_spp_inst >= 0.5

        refine_proposals_pred = torch.gather(spp_mask, dim=-1, index=spp_ids.expand(n_inst, n_points))

        return refine_proposals_pred

    if len(proposals_pred.shape) == 1:
        proposals_pred = proposals_pred.float()
        spp_unique, spp_ids = torch.unique(spp, return_inverse=True)

        mean_spp_inst = torch_scatter.scatter(proposals_pred.float(), spp_ids, dim=-1, reduce="mean")  # n_inst, n_spp
        spp_mask = mean_spp_inst >= 0.5

        refine_proposals_pred = spp_mask[spp_ids]
        refine_proposals_pred = refine_proposals_pred.bool()
        return refine_proposals_pred


def gen_boundary_gt(
    semantic_labels,
    instance_labels,
    coords_float,
    batch_idxs,
    radius=0.2,
    neighbor=48,
    ignore_label=255,
    label_shift=2,
):
    boundary = torch.zeros((instance_labels.shape[0]), dtype=torch.float, device=instance_labels.device)
    # condition = (instance_labels != ignore_label) & (semantic_labels >= label_shift)

    # condition = (semantic_labels >= 0)
    condition = torch.ones_like(semantic_labels).bool()
    object_idxs = torch.nonzero(condition).view(-1)

    if len(object_idxs) == 0:
        return boundary

    coords_float_ = coords_float[object_idxs]
    instance_ = instance_labels[object_idxs][:, None]

    batch_size = len(torch.unique(batch_idxs))
    batch_offsets = get_batch_offsets(batch_idxs, batch_size)
    batch_idxs_ = batch_idxs[object_idxs]
    batch_offsets_ = get_batch_offsets(batch_idxs_, batch_size)
    # batch_offsets_ = torch.tensor([0, coords_float_.shape[0]], dtype=torch.int).cuda()

    neighbor_inds = ballquery_batchflat(radius, neighbor, coords_float, coords_float_, batch_offsets, batch_offsets_)
    # neighbor_inds, _ = knnquery(neighbor, coords_float, coords_float_, batch_offsets, batch_offsets_)

    # print(neighbor_inds.shape, coords_float.shape)
    neighbor_inds = neighbor_inds.view(-1).long()
    neighbor_instance = instance_labels[neighbor_inds].view(coords_float_.shape[0], neighbor)

    diff_ins = torch.any((neighbor_instance != instance_), dim=-1)  # mpoints
    # boundary_labels[object_idxs.long()] = (diff_sem | diff_ins).long()

    # boundary_labels = boundary_labels.cpu()

    boundary[object_idxs.long()] = diff_ins.float()

    return boundary

def cartesian_to_spherical( xyz, device='cuda'):
    if len(xyz.shape) == 2:
        if device == 'cuda':
            
            ptsnew = torch.hstack((xyz, torch.zeros(xyz.shape).cuda()))
            
        else:
            ptsnew = torch.hstack((xyz, torch.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = torch.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,4] = torch.arctan2(torch.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,5] = torch.arctan2(xyz[:,1], xyz[:,0])
    elif len(xyz.shape) == 3:
        
        ptsnew = torch.cat([xyz, torch.zeros(xyz.shape).cuda()],2)
        xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
        ptsnew[:,:,3] = torch.sqrt(xy + xyz[:,:,2]**2)
        ptsnew[:,:,4] = torch.arctan2(torch.sqrt(xy), xyz[:,:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,:,5] = torch.arctan2(xyz[:,:,1], xyz[:,:,0])
    elif len(xyz.shape) == 4:
        
        ptsnew = torch.cat([xyz, torch.zeros(xyz.shape).float().cuda()],3)
        xy = xyz[:,:,:,0]**2 + xyz[:,:,:,1]**2
        ptsnew[:,:,:,3] = torch.sqrt(xy + xyz[:,:,:,2]**2)
        ptsnew[:,:,:,4] = torch.arctan2(torch.sqrt(xy), xyz[:,:,:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,:,:,5] = torch.arctan2(xyz[:,:,:,1], xyz[:,:,:,0])
    return ptsnew
def spherical_to_cartesian(spherical_points):
    
    if len(spherical_points.shape) == 2:
        rho, theta, phi = spherical_points[:,0],spherical_points[:,1],spherical_points[:,2]
    elif len(spherical_points.shape) == 3:
        rho, theta, phi = spherical_points[:,:,0],spherical_points[:,:,1],spherical_points[:,:,2]
    elif len(spherical_points.shape) == 4:
        rho, theta, phi = spherical_points[:,:,:,0],spherical_points[:,:,:,1],spherical_points[:,:,:,2]
   
    x = rho * torch.sin(theta) * torch.cos(phi)
    y = rho * torch.sin(theta) * torch.sin(phi)
    z = rho * torch.cos(theta)
    if len(spherical_points.shape) == 2:
        return torch.cat([x[:,None],y[:,None],z[:,None]],1)
    elif len(spherical_points.shape) == 3:
        return torch.cat([x[:,:,None],y[:,:,None],z[:,:,None]],2)
    elif len(spherical_points.shape) == 4:
        return torch.cat([x[:,:,:,None],y[:,:,:,None],z[:,:,:,None]],3)
    

def divide_with_angle_vectorize_3d( spherical_points, num_ray_width, num_ray_height, rays, return_ref=False, cartesian_ref=None):
    
    #spherical_points : MXNX3
    #num_ray_width, num_ray_height: (),()
    points_dict = {}
    dist_labels = torch.zeros(spherical_points.shape[0],num_ray_width * num_ray_height).to(spherical_points.device)
    dist_labels[:] = 1e-2
    
    dist_labels_from_anchors = torch.zeros(spherical_points.shape[0],(num_ray_width * num_ray_height)).to(spherical_points.device)
    dist_labels_from_anchors[:] = 1e-2
    
    
    cnt = 0
   
   
    unit_h = 180/num_ray_height
    unit_w = 360/num_ray_width
    
    
    xy_angles = torch.arange(0,360, int(360/num_ray_width)).to(spherical_points.device)
    yz_angles = torch.arange(0,180, int(180/num_ray_height)).to(spherical_points.device)
    
    diff_yz = spherical_points[:,:,1][:,:,None].repeat(1,1,len(yz_angles)) - yz_angles[None,None,:]
    diff_yz[diff_yz < 0 ] = unit_h + 1
    diff_yz = diff_yz / unit_h
    yz_min_val, yz_angles_num = torch.min(diff_yz,2)    



    diff_xy = spherical_points[:,:,2][:,:,None].repeat(1,1,len(xy_angles)) - xy_angles[None,None,:]
    diff_xy[diff_xy < 0 ] = unit_w + 1
    diff_xy = diff_xy / unit_w
    xy_min_val, xy_angles_num = torch.min(diff_xy, 2)
    
    
    angles_num = yz_angles_num*len(xy_angles) + xy_angles_num 
    #for idx in range(np.amax([num_ray_width,num_ray_height])):
    
    #    if num_ray_width > idx :
    points_bag = torch.zeros(angles_num.shape[0], angles_num.shape[1] ,num_ray_height*num_ray_width).to(spherical_points.device)
    points_bag[:,:,:] = 0.01 ### To check if there's any point that are not assigned

    angles_num_ravel = angles_num.reshape(-1)
    samples_num = torch.arange(len(angles_num_ravel))
    samples_num = samples_num / angles_num.shape[1]
    
    samples_num = torch.clamp(samples_num, 0, angles_num.shape[0]-1)
    samples_num = samples_num.long()

    points_bag_ravel = points_bag.reshape(-1,num_ray_height*num_ray_width)
    spherical_points_ravel = spherical_points[:,:,0].reshape(-1)


    

    
    points_bag_ravel[torch.arange(len(angles_num_ravel)).long(), angles_num_ravel] = spherical_points_ravel 
   
    
    #if samples_num.max().item() >= rays.shape[0] or angles_num_ravel.max().item() >= rays.shape[1]:
    #    pdb.set_trace()
    rays_ravel = rays[samples_num, angles_num_ravel]##problem
    
    rays_ravel = rays_ravel.reshape(points_bag.shape[0], points_bag.shape[1])
    
    points_bag = points_bag_ravel.reshape(points_bag.shape)
    points_in_ray,_ = torch.max(points_bag,2)

    
    inside_ray = (points_in_ray <= rays_ravel) *  ( rays_ravel > 0.01)
    
    inside_locs_roi = torch.where(inside_ray)
    
    #pdb.set_trace()
    points_bag = points_bag.permute(0,2,1)
    dist_anchors, _ = torch.max(points_bag,2)

    ###########
    #dist_anchors = dist_anchors*1.3
    ###########
    #argmax = torch.argmax(points_bag,0)

    

    return dist_anchors, angles_num, inside_locs_roi


def divide_with_angle_vectorize_3d_gamma( spherical_points, num_ray_width, num_ray_height, rays, return_ref=False, cartesian_ref=None):
    
    #spherical_points : MXNX3
    #num_ray_width, num_ray_height: (),()
    points_dict = {}
    dist_labels = torch.zeros(spherical_points.shape[0],num_ray_width * num_ray_height).to(spherical_points.device)
    dist_labels[:] = 1e-2
    
    dist_labels_from_anchors = torch.zeros(spherical_points.shape[0],(num_ray_width * num_ray_height)).to(spherical_points.device)
    dist_labels_from_anchors[:] = 1e-2
    
   
    unit_h = 180/num_ray_height
    unit_w = 360/num_ray_width
    
    
    xy_angles = torch.arange(0,360, int(360/num_ray_width)).to(spherical_points.device)
    yz_angles = torch.arange(0,180, int(180/num_ray_height)).to(spherical_points.device)
    
    diff_yz = spherical_points[:,:,1][:,:,None].repeat(1,1,len(yz_angles)) - yz_angles[None,None,:]
    diff_yz[diff_yz < 0 ] = unit_h + 1
    diff_yz = diff_yz / unit_h
    yz_min_val, yz_angles_num = torch.min(diff_yz,2)    



    diff_xy = spherical_points[:,:,2][:,:,None].repeat(1,1,len(xy_angles)) - xy_angles[None,None,:]
    diff_xy[diff_xy < 0 ] = unit_w + 1
    diff_xy = diff_xy / unit_w
    xy_min_val, xy_angles_num = torch.min(diff_xy, 2)
    
    
    angles_num = yz_angles_num*len(xy_angles) + xy_angles_num 
    #for idx in range(np.amax([num_ray_width,num_ray_height])):
    
    #    if num_ray_width > idx :
    points_bag = torch.zeros(angles_num.shape[0], angles_num.shape[1] ,num_ray_height*num_ray_width).to(spherical_points.device)
    points_bag[:,:,:] = -100.0#0.01 ### To check if there's any point that are not assigned

    angles_num_ravel = angles_num.reshape(-1)
    samples_num = torch.arange(len(angles_num_ravel))
    samples_num = samples_num / angles_num.shape[1]
    
    samples_num = torch.clamp(samples_num, 0, angles_num.shape[0]-1)
    samples_num = samples_num.long()

    points_bag_ravel = points_bag.reshape(-1,num_ray_height*num_ray_width)
    spherical_points_ravel = spherical_points[:,:,0].reshape(-1)



    points_bag_ravel[torch.arange(len(angles_num_ravel)).long(), angles_num_ravel] = spherical_points_ravel 
   
    
    #if samples_num.max().item() >= rays.shape[0] or angles_num_ravel.max().item() >= rays.shape[1]:
    #    pdb.set_trace()
    rays_ravel = rays[samples_num, angles_num_ravel]##problem
    
    rays_ravel = rays_ravel.reshape(points_bag.shape[0], points_bag.shape[1])
    
    points_bag = points_bag_ravel.reshape(points_bag.shape)
    points_in_ray,_ = torch.max(points_bag,2)

    
    inside_ray = (points_in_ray <= rays_ravel)# *  ( rays_ravel > 0.01)
    
    inside_locs_roi = torch.where(inside_ray)
    
    
    points_bag = points_bag.permute(0,2,1)
    dist_anchors, _ = torch.max(points_bag,2)

    ###########
  


    return dist_anchors, angles_num, inside_locs_roi

def find_sector_( spherical_points, num_ray_width, num_ray_height, rays, return_ref=False, cartesian_ref=None):
    
    #spherical_points : MXNX3
    #num_ray_width, num_ray_height: (),()
    
    dist_labels = torch.zeros(spherical_points.shape[0],num_ray_width * num_ray_height).to(spherical_points.device)
    dist_labels[:] = 1e-2
    
    dist_labels_from_anchors = torch.zeros(spherical_points.shape[0],(num_ray_width * num_ray_height)).to(spherical_points.device)
    dist_labels_from_anchors[:] = 1e-2
    
    unit_h = 180/num_ray_height
    unit_w = 360/num_ray_width
    
    
    xy_angles = torch.arange(0,360, int(360/num_ray_width)).to(spherical_points.device)
    yz_angles = torch.arange(0,180, int(180/num_ray_height)).to(spherical_points.device)
    
    diff_yz = spherical_points[:,:,1][:,:,None].repeat(1,1,len(yz_angles)) - yz_angles[None,None,:]
    diff_yz[diff_yz < 0 ] = unit_h + 1
    diff_yz = diff_yz / unit_h
    yz_min_val, yz_angles_num = torch.min(diff_yz,2)    



    diff_xy = spherical_points[:,:,2][:,:,None].repeat(1,1,len(xy_angles)) - xy_angles[None,None,:]
    diff_xy[diff_xy < 0 ] = unit_w + 1
    diff_xy = diff_xy / unit_w
    xy_min_val, xy_angles_num = torch.min(diff_xy, 2)
    
    
    angles_num = yz_angles_num*len(xy_angles) + xy_angles_num 
    
    points_bag = torch.zeros(angles_num.shape[0], angles_num.shape[1] ,num_ray_height*num_ray_width).to(spherical_points.device)
    points_bag[:,:,:] = -100.0

    angles_num_ravel = angles_num.reshape(-1)
    samples_num = torch.arange(len(angles_num_ravel))
    samples_num = samples_num / angles_num.shape[1]
    
    samples_num = torch.clamp(samples_num, 0, angles_num.shape[0]-1)
    samples_num = samples_num.long()

    points_bag_ravel = points_bag.reshape(-1,num_ray_height*num_ray_width)
    spherical_points_ravel = spherical_points[:,:,0].reshape(-1)



    points_bag_ravel[torch.arange(len(angles_num_ravel)).long(), angles_num_ravel] = spherical_points_ravel 
   
    rays_ravel = rays[samples_num, angles_num_ravel]
    rays_ravel = rays_ravel.reshape(points_bag.shape[0], points_bag.shape[1])
    
    points_bag = points_bag_ravel.reshape(points_bag.shape)
    points_in_ray,_ = torch.max(points_bag,2)

    
    inside_ray = (points_in_ray <= rays_ravel)
    inside_locs_roi = torch.where(inside_ray)
    
    
    points_bag = points_bag.permute(0,2,1)
    dist_anchors, _ = torch.max(points_bag,2)

    return dist_anchors, angles_num, inside_locs_roi

def divide_with_angle_vectorize_3d_ray( spherical_points, num_ray_width, num_ray_height, return_ref=False, cartesian_ref=None):
    
    #spherical_points : MXNX3
    #num_ray_width, num_ray_height: (),()
    points_dict = {}
    dist_labels = torch.zeros(spherical_points.shape[0],num_ray_width * num_ray_height).to(spherical_points.device)
    dist_labels[:] = 1e-2
    
    dist_labels_from_anchors = torch.zeros(spherical_points.shape[0],(num_ray_width * num_ray_height)).to(spherical_points.device)
    dist_labels_from_anchors[:] = 1e-2
    
    
    cnt = 0
   
   
    unit_h = 180/num_ray_height
    unit_w = 360/num_ray_width
    
    
    xy_angles = torch.arange(0,360, int(360/num_ray_width)).to(spherical_points.device)
    yz_angles = torch.arange(0,180, int(180/num_ray_height)).to(spherical_points.device)
    
    diff_yz = spherical_points[:,:,1][:,:,None].repeat(1,1,len(yz_angles)) - yz_angles[None,None,:]
    diff_yz[diff_yz < 0 ] = unit_h + 1
    diff_yz = diff_yz / unit_h
    yz_min_val, yz_angles_num = torch.min(diff_yz,2)    



    diff_xy = spherical_points[:,:,2][:,:,None].repeat(1,1,len(xy_angles)) - xy_angles[None,None,:]
    diff_xy[diff_xy < 0 ] = unit_w + 1
    diff_xy = diff_xy / unit_w
    xy_min_val, xy_angles_num = torch.min(diff_xy, 2)
    
    
    angles_num = yz_angles_num*len(xy_angles) + xy_angles_num 
    #for idx in range(np.amax([num_ray_width,num_ray_height])):
    
    #    if num_ray_width > idx :
    points_bag = torch.zeros(angles_num.shape[0], angles_num.shape[1] ,num_ray_height*num_ray_width).to(spherical_points.device)
    points_bag[:,:,:] = 1e-2 ### To check if there's any point that are not assigned

    angles_num_ravel = angles_num.reshape(-1)
    samples_num = torch.arange(len(angles_num_ravel))
    samples_num = samples_num / angles_num.shape[1]
    samples_num = samples_num.long()

    points_bag_ravel = points_bag.reshape(-1,num_ray_height*num_ray_width)
    spherical_points_ravel = spherical_points[:,:,0].reshape(-1)





    points_bag_ravel[torch.arange(len(angles_num_ravel)).long(), angles_num_ravel] = spherical_points_ravel 
    
    
    
    points_bag = points_bag_ravel.reshape(points_bag.shape)
   
    points_in_ray,_ = torch.max(points_bag,2)

    
   
    
    #pdb.set_trace()
    points_bag = points_bag.permute(0,2,1)
    dist_anchors, _ = torch.max(points_bag,2)

    ###########
    #dist_anchors = dist_anchors*1.3
    ###########
    #argmax = torch.argmax(points_bag,0)

    return dist_anchors, angles_num
def divide_with_angle_vectorize( spherical_points, num_ray_width, num_ray_height, return_ref=False, cartesian_ref=None):
    
    #spherical_points : NX3
    #num_ray_width, num_ray_height: (),()
    points_dict = {}
    dist_labels = torch.zeros(num_ray_width * num_ray_height).to(spherical_points.device)
    dist_labels[:] = 1e-6
    
    dist_labels_from_anchors = torch.zeros((num_ray_width * num_ray_height)).to(spherical_points.device)
    dist_labels_from_anchors[:] = 1e-6
    
    
    cnt = 0
   
   
    unit_h = 180/num_ray_height
    unit_w = 360/num_ray_width
    
    
    xy_angles = torch.arange(0,360, int(360/num_ray_width)).to(spherical_points.device)
    yz_angles = torch.arange(0,180, int(180/num_ray_height)).to(spherical_points.device)
    
    diff_yz = spherical_points[:,1][:,None].repeat(1,len(yz_angles)) - yz_angles[None,:]
    diff_yz[diff_yz < 0 ] = unit_h + 1
    diff_yz = diff_yz / unit_h
    yz_min_val, yz_angles_num = torch.min(diff_yz,1)    



    diff_xy = spherical_points[:,2][:,None].repeat(1,len(xy_angles)) - xy_angles[None,:]
    diff_xy[diff_xy < 0 ] = unit_w + 1
    diff_xy = diff_xy / unit_w
    xy_min_val, xy_angles_num = torch.min(diff_xy, 1)
    
    
    angles_num = yz_angles_num*len(xy_angles) + xy_angles_num 
    #for idx in range(np.amax([num_ray_width,num_ray_height])):
    
    #    if num_ray_width > idx :
    points_bag = torch.zeros(len(angles_num) ,num_ray_height*num_ray_width).to(spherical_points.device)
    points_bag[:,:] = 1e-6

    points_bag[torch.arange(len(angles_num)).long(), angles_num.long() ] = spherical_points[:,0].float()
    #pdb.set_trace()
    points_bag = points_bag.T
    dist_anchors, _ = torch.max(points_bag,1)
    
    ###########
    #dist_anchors = dist_anchors*1.3
    ###########
    #argmax = torch.argmax(points_bag,0)

    

    return dist_anchors, angles_num
def get_angular_gt(inst_points, center):
    

    pointwise_centered = center[None,:] - inst_points
    pointwise_in_sph = cartesian_to_spherical(pointwise_centered)

    angles_in_rad = pointwise_in_sph[:,4:]
    return angles_in_rad
    


def get_sph_gt(inst_points, inst_points_center, rays_width, rays_height, angles_ref, centering=False, chk_iou=True, all_points=None, inst_loc=None, debug=False):
    
    #if centering:
    #    centered_points = inst_points - inst_points_center[None,:]
    #else:
    #    centered_points = inst_points
    #pointwise_angular_gt = get_angular_gt(inst_points, inst_points_center)
    pointwise_angular_gt = inst_points[:,:2]
    #pdb.set_trace()
    centered_points = inst_points - inst_points_center[None,:]
    new_pts = cartesian_to_spherical(centered_points)
    
    new_pts[:,4:] = torch.rad2deg(new_pts[:,4:])
    new_pts[:,5] += 180
    
    #start_time = time.time()
    dist_label_anchors, angles_num_each = divide_with_angle_vectorize(new_pts[:,3:], rays_width, rays_height, return_ref=True, cartesian_ref=inst_points)
    #print('time vec : {0}'.format(time.time() - start_time))
    if debug:
        real_points = []
        for idx_debug in range(rays_width*rays_height):
            
            angle_mask = angles_num_each == idx_debug
            points_inside_angles = new_pts[:,3:][angle_mask]
            points_cart = inst_points[angle_mask]

            if len(points_inside_angles) > 0:
                furthest = torch.amax(points_inside_angles[:,0])
                furthest_arg = torch.argmax(points_inside_angles[:,0])
                if furthest != dist_label_anchors[idx_debug]:
                    print('err')
                    pdb.set_trace()
                furthest_cart = points_cart[furthest_arg].cpu().numpy()
            else:
                furthest_cart = np.zeros(3)
            real_points.append(furthest_cart)   
        
        angles_num_each_np = angles_num_each.cpu().numpy()
        real_points = np.array(real_points)
    if chk_iou: # validate upper bound of gt in IoU 
        
        inst_mask = torch.zeros(len(all_points)).cuda()
        inst_mask[inst_loc] = 1
        inst_mask = inst_mask == 1
        gt_mask_pure, gt_mask_angle = torch.zeros(len(all_points)).cuda(), torch.zeros(len(all_points)).cuda()
        
        angular_field_all = torch.zeros(len(all_points),2).cuda()
        angular_field_all[:,:] = -100
        angular_field_all[inst_loc] = pointwise_angular_gt


        centered_points_all = all_points - inst_points_center[None,:]
        new_pts_all = cartesian_to_spherical(centered_points_all)
        new_pts_all[:,4:] = torch.rad2deg(new_pts_all[:,4:])
        new_pts_all[:,5] += 180

        
        points_inside_loc = get_points_inside_s_contour_vectorize(dist_label_anchors, new_pts_all[:,3:], rays_width, rays_height)
        gt_mask_pure[points_inside_loc] = 1
        gt_mask_pure = gt_mask_pure == 1

        points_inside = all_points[points_inside_loc]
        
        
        angular_gt_inside_pure_mask = get_angular_gt(points_inside, inst_points_center)
        angular_gt_est = angular_field_all[points_inside_loc]

        sim_score = F.cosine_similarity(angular_gt_inside_pure_mask, angular_gt_est)
        sim_mask = sim_score > 0.9
        points_inside_loc_angular = points_inside_loc[sim_mask]
        gt_mask_angle[points_inside_loc_angular] = 1
        gt_mask_angle = gt_mask_angle == 1

        intersection_pure_ray = gt_mask_pure * inst_mask
        intersection_angular = gt_mask_angle * inst_mask

        iou_pure_ray = intersection_pure_ray.sum() / (gt_mask_pure.sum() + inst_mask.sum() - intersection_pure_ray.sum())
        iou_angular = intersection_angular.sum() / (gt_mask_angle.sum() + inst_mask.sum() - intersection_angular.sum())

    
    if centering == False:
       
        sph_coords = torch.cat([dist_label_anchors[:,None], angles_ref.to(dist_label_anchors.device) ],1)
        sph_coords[:,2] -= 180
        sph_coords[:,1:] = torch.deg2rad(sph_coords[:,1:])
        sph_points = spherical_to_cartesian(sph_coords)
        sph_points += inst_points_center[None,:]
        if debug:
            
            return sph_points, pointwise_angular_gt, [iou_pure_ray.item(), iou_angular.item()], dist_label_anchors[:,None], real_points, angles_num_each_np
        if chk_iou:
            return sph_points, pointwise_angular_gt, [iou_pure_ray.item(), iou_angular.item()], dist_label_anchors[:,None]
        
        return sph_points, pointwise_angular_gt, [], dist_label_anchors[:,None]
    
    return dist_label_anchors, angles_num_each


def get_mask_inside_ray(dists_points, dist_ray, angles_num, num_rays, num_range):
    #dists_points : N,
    #dist_ray: num_rays(height*width)
    #angles: N
    
    mask_rays = torch.zeros(num_rays,num_range).float().cuda()
    
    for idx_ray in range(num_rays):
        #if len(dists) != len(angles_num_each):
        
        valid_points_in_ray = dists_points[angles_num==idx_ray]
        
        ray_dist = dist_ray[idx_ray]
       
        if len(valid_points_in_ray) > 0:
            if ray_dist == 0:
                label = torch.zeros(num_range).float().cuda()
                label[0] = 1
                mask_rays[idx_ray] = label
            else:
                step_val = ray_dist/(num_range)
                
                range_values = torch.arange(0.0, ray_dist, step_val).float().cuda()
                if len(range_values) > num_range:
                    range_values = range_values[:num_range]
                range_max = valid_points_in_ray[:,None] - range_values[None,:]
                range_max[range_max < 0] = 1000000000
                
                range_labels = torch.argmin(range_max,1)
                
                label = torch.zeros(num_range).float().cuda()
                label[range_labels] = 1
                mask_rays[idx_ray] = label
    return mask_rays

def get_mask_indice_inside_ray(dists_points, dist_ray, rays_inside_pred, angles_num, num_rays, num_range):
    #dists_points : N,
    #dist_ray: num_rays(height*width)
    #angles: N
    
    mask_rays = torch.zeros(num_rays,num_range).float().cuda()
    mask_points_indice = []
    total_point_num_inside_ray = 0
    total_selected_points = 0
    for idx_ray in range(num_rays):
        #if len(dists) != len(angles_num_each):
        
        surv_indice = torch.where(angles_num == idx_ray)[0]

        valid_points_in_ray = dists_points[angles_num==idx_ray]
        
        ray_dist = dist_ray[idx_ray]       
        sec_mask = valid_points_in_ray < ray_dist
        surv_indice = surv_indice[sec_mask]
        valid_points_in_ray = valid_points_in_ray[sec_mask]

        if len(valid_points_in_ray) > 0:
            total_point_num_inside_ray += len(valid_points_in_ray)
            if ray_dist == 0:
                label = torch.zeros(num_range).float().cuda()
                label[0] = 1
                mask_rays[idx_ray] = label
            else:
                step_val = ray_dist/(num_range)
                
                range_values = torch.arange(0.0, ray_dist, step_val).float().cuda()
                if len(range_values) > num_range:
                    range_values = range_values[:num_range]
                range_max = valid_points_in_ray[:,None] - range_values[None,:]
                range_max[range_max < 0] = 1000000000
                
                range_labels = torch.argmin(range_max,1)
                


                label = torch.zeros(num_range).float().cuda()
                label[range_labels] = 1
                pred = rays_inside_pred[idx_ray]
                pred = (pred > 0) * 1.0 
                for idx_range in range(pred.shape[0]):
                    if pred[idx_range] == 1:
                        range_mask = range_labels == idx_range
                        if range_mask.sum().item() > 0:
                            total_selected_points += range_mask.sum().item()
                            points_num = surv_indice[range_mask]
                            mask_points_indice.append(points_num)
              
    if len(mask_points_indice) > 0:
        mask_points_indice = torch.cat(mask_points_indice)
    return mask_points_indice

def get_instance_info_dyco_ray(coords_float, batch_idxs, instance_labels, semantic_labels, rays_width, rays_height, angles_ref, label_shift=2, is_analyze=False, is_vis=False):
    instance_pointnum = []
    instance_cls = []
    instance_box = []
    instance_sphere = []
    instance_num = int(instance_labels.max()) + 1

    centroid_offset_labels = (
        torch.ones((coords_float.shape[0], 3), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )
    angular_labels = (
        torch.ones((coords_float.shape[0], 2), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )
    corners_offset_labels = (
        torch.ones((coords_float.shape[0], 3 * 2), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )

    spherical_labels = (
        torch.ones((coords_float.shape[0], (rays_width * rays_height) + 1, 3), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )
    spherical_mask = (
        torch.zeros((coords_float.shape[0]), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )    


   
    centered_coords = coords_float
    
    rays_gt_list = []
    num_pos_objs = 6
    if is_vis:
        real_points_list = []
        angles_num_list = []
    
    for i_ in range(instance_num):
       
        inst_idx_i = torch.nonzero(instance_labels == i_).view(-1)
        coords_float_i = coords_float[inst_idx_i]
        coords_float_i_centered = centered_coords[inst_idx_i]
        
        centroid = coords_float_i.mean(dim=0)
        centroid_centered = coords_float_i_centered.mean(dim=0)
        sem_labels = semantic_labels[inst_idx_i]
       
        if is_vis:
            sph_label, pointwise_angular_gt, ious_upper_bound, rays_gt, real_points, angles_num = get_sph_gt( coords_float_i_centered, centroid_centered, rays_width, rays_height, 
                                                     angles_ref, chk_iou=True, all_points=coords_float, inst_loc=inst_idx_i, debug=is_vis) # Getting the cartesian coords(centered) of sphere for each predefined labels
            real_points_list.append(real_points)
            angles_num_list.append(angles_num)
        else:
         
            sph_label, pointwise_angular_gt, ious_upper_bound, rays_gt = get_sph_gt( coords_float_i_centered, centroid_centered, rays_width, rays_height, 
                                                        angles_ref, chk_iou=False, all_points=coords_float, inst_loc=inst_idx_i, debug=is_vis) # Getting the cartesian coords(centered) of sphere for each predefined labels
       
        rays_gt = torch.cat([rays_gt, centroid[:,None]],0)
       
        min_xyz_i = coords_float_i.min(dim=0)[0]
        max_xyz_i = coords_float_i.max(dim=0)[0]
        
        centers_mask = torch.zeros(len(coords_float_i)).float().cuda()
        dist_to_center = torch.cdist(centroid[None,:], coords_float_i)[0]
        dist_sorted = torch.argsort(dist_to_center)
        sem_sorted = sem_labels[dist_sorted]
        sem_mask = sem_sorted != -100
        dist_sorted = dist_sorted[sem_mask]
        
        centers_mask[dist_sorted[:num_pos_objs]] = 1

        angular_labels[inst_idx_i] = pointwise_angular_gt
        centroid_offset_labels[inst_idx_i] = centroid - coords_float_i
        corners_offset_labels[inst_idx_i, 0:3] = min_xyz_i - coords_float_i
        corners_offset_labels[inst_idx_i, 3:6] = max_xyz_i - coords_float_i

       
        
        sph_shift_center = centroid[None,:] - coords_float_i_centered
        spherical_labels[inst_idx_i, :-1,:] = rays_gt[:-3]
        spherical_labels[inst_idx_i, -1] = sph_shift_center
        spherical_mask[inst_idx_i] = centers_mask
        
        
        sph_label = torch.cat([sph_label, centroid[None,:]],0)

        instance_box.append(torch.cat([min_xyz_i, max_xyz_i], axis=0))
        instance_sphere.append(sph_label)
        instance_pointnum.append(len(inst_idx_i))
        cls_idx = inst_idx_i[0]
        
        instance_cls.append(semantic_labels[cls_idx].clone())
        
        rays_gt_list.append(rays_gt)
    
    instance_cls = torch.tensor(instance_cls, device=coords_float.device)
    instance_box = torch.stack(instance_box, dim=0)  # N, 6
    
    rays_gt_list = torch.stack(rays_gt_list,0)
    
    instance_sphere = torch.stack(instance_sphere, dim=0)
    instance_cls[instance_cls != -100] = instance_cls[instance_cls != -100] - label_shift
 
    if is_vis:
        
        return instance_cls, instance_box, instance_sphere, centroid_offset_labels, angular_labels, corners_offset_labels, spherical_labels, rays_gt_list, spherical_mask, semantic_labels, real_points_list
    return instance_cls, instance_box, instance_sphere, centroid_offset_labels, angular_labels, corners_offset_labels, spherical_labels, rays_gt_list, spherical_mask, semantic_labels#corners_offset_labels

def get_inliers(dc_coords_float, mask_all, rays, rays_width, rays_height):
    
    centers = rays[:,-3:]
    centered_coords = dc_coords_float[None,:] - centers[:,None,:]
    new_pts_vec = cartesian_to_spherical(centered_coords)
    new_pts_vec[:,:,3] = new_pts_vec[:,:,3] - mask_all 
    
    
    new_pts_vec[:,:,4:] = torch.rad2deg(new_pts_vec[:,:,4:])
    new_pts_vec[:,:,5] += 180

    
    dist_anchors, angles_num, roi_locs = divide_with_angle_vectorize_3d_gamma(new_pts_vec[:,:,3:], rays_width, rays_height, rays)
   
    return dist_anchors, angles_num, roi_locs, new_pts_vec

def get_3D_locs_from_rays(rays, ref_angles, centers):
    #ref_angles should be in angles(degree): NXrays_width*rays_height
    #angles_ref: (rays_width*rays_height X 2)
    #center: (rays_width*rays_height X 3)
    
    ref_angles = ref_angles[None,:,:].repeat(len(rays),1,1)

    
    sph_coords = torch.cat([rays[:,:,None], ref_angles.to(rays.device) ],2)
    sph_coords[:,:,2] -= 180
    sph_coords[:,:,1:] = torch.deg2rad(sph_coords[:,:,1:])
    sph_points = spherical_to_cartesian(sph_coords)
    sph_points += centers[:,None,:]

    return sph_points

def get_instance_info(coords_float, instance_labels, semantic_labels, label_shift=2):
    instance_pointnum = []
    instance_cls = []
    instance_box = []
    instance_num = int(instance_labels.max()) + 1

    centroid_offset_labels = (
        torch.ones((coords_float.shape[0], 3), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )
    corners_offset_labels = (
        torch.ones((coords_float.shape[0], 3 * 2), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )

    for i_ in range(instance_num):
        inst_idx_i = torch.nonzero(instance_labels == i_).view(-1)
        coords_float_i = coords_float[inst_idx_i]

        centroid = coords_float_i.mean(dim=0)

        min_xyz_i = coords_float_i.min(dim=0)[0]
        max_xyz_i = coords_float_i.max(dim=0)[0]

        centroid_offset_labels[inst_idx_i] = centroid - coords_float_i
        corners_offset_labels[inst_idx_i, 0:3] = min_xyz_i - coords_float_i
        corners_offset_labels[inst_idx_i, 3:6] = max_xyz_i - coords_float_i
        
        instance_box.append(torch.cat([min_xyz_i, max_xyz_i], axis=0))

        instance_pointnum.append(len(inst_idx_i))
        cls_idx = inst_idx_i[0]
        instance_cls.append(semantic_labels[cls_idx])
    
    instance_cls = torch.tensor(instance_cls, device=coords_float.device)
    instance_box = torch.stack(instance_box, dim=0)  # N, 6
    instance_cls[instance_cls != -100] = instance_cls[instance_cls != -100] - label_shift
    
    return instance_cls, instance_box, centroid_offset_labels, corners_offset_labels


def get_batch_offsets(batch_idxs, bs):
    batch_offsets = torch.zeros((bs + 1), dtype=torch.int, device=batch_idxs.device)
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

def random_downsample(batch_offsets, batch_size, n_subsample=30000):
    idxs_subsample = []
    for b in range(batch_size):
        start, end = batch_offsets[b], batch_offsets[b + 1]
        num_points_b = (end - start).cpu()

        if n_subsample == -1 or n_subsample >= num_points_b:
            new_inds = torch.arange(num_points_b, dtype=torch.long, device=batch_offsets.device) + start
        else:
            new_inds = (
                torch.tensor(
                    np.random.choice(num_points_b, n_subsample, replace=False),
                    dtype=torch.long,
                    device=batch_offsets.device,
                )
                + start
            )
        idxs_subsample.append(new_inds)
    idxs_subsample = torch.cat(idxs_subsample)  # N_subsample: batch x 20000

    return idxs_subsample


def get_cropped_instance_label(instance_label, valid_idxs=None):
    if valid_idxs is not None:
        instance_label = instance_label[valid_idxs]
    j = 0
    while j < instance_label.max():
        if (instance_label == j).sum() == 0:
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label


def custom_scatter_mean(input_feats, indices, dim=0, pool=True, output_type=None):
    if not pool:
        return input_feats

    original_type = input_feats.dtype
    with torch.cuda.amp.autocast(enabled=False):
        out_feats = torch_scatter.scatter_mean(input_feats.to(torch.float32), indices, dim=dim)

    if output_type is None:
        out_feats = out_feats.to(original_type)
    else:
        out_feats = out_feats.to(output_type)

    return out_feats


def superpoint_major_voting(
    labels, superpoint, n_classes, has_ignore_label=False, ignore_label=-100, return_full=True
):
    if has_ignore_label:
        labels = torch.where(labels >= 0, labels + 1, 0)
        n_classes += 1

    n_points = len(labels)
    # semantic_preds = voting_semantic_segmentation(semantic_preds, superpoint, num_semantic=self.classes)
    onehot_semantic_preds = F.one_hot(labels.long(), num_classes=n_classes)

    # breakpoint()
    count_onehot_semantic_preds = torch_scatter.scatter(
        onehot_semantic_preds, superpoint[:, None].expand(n_points, n_classes), dim=0, reduce="sum"
    )  # n_labels, n_spp

    label_spp = torch.argmax(count_onehot_semantic_preds, dim=1)  # n_spp
    score_label_spp = count_onehot_semantic_preds / torch.sum(count_onehot_semantic_preds, dim=1, keepdim=True)

    if has_ignore_label:
        label_spp = torch.where(label_spp >= 1, label_spp - 1, ignore_label)

    if return_full:
        refined_labels = label_spp[superpoint]
        refine_scores = score_label_spp[superpoint]

        return refined_labels, refine_scores

    return label_spp, score_label_spp


def get_subsample_gt(
    instance_labels, subsample_idxs, instance_cls, instance_box, subsample_batch_offsets, batch_size, ignore_label=-100
):
    subsample_inst_mask_arr = []

    subsample_instance_labels = instance_labels[subsample_idxs]
    for b in range(batch_size):
        start, end = subsample_batch_offsets[b], subsample_batch_offsets[b + 1]
        n_subsample = end - start

        instance_labels_b = subsample_instance_labels[start:end]

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != ignore_label]
       
        unique_inst = unique_inst[instance_cls[unique_inst] >= 0]

        n_inst_gt = len(unique_inst)

        if n_inst_gt == 0:
            subsample_inst_mask_arr.append(None)
            continue

        instance_cls_b = instance_cls[unique_inst]
        instance_box_b = instance_box[unique_inst]

        subsample_mask_labels_b = torch.zeros(
            (n_inst_gt, n_subsample), device=subsample_instance_labels.device, dtype=torch.float
        )
        # breakpoint()
        for i, uni_id in enumerate(unique_inst):
            mask_ = instance_labels_b == uni_id
            subsample_mask_labels_b[i] = mask_.float()

        subsample_inst_mask_arr.append(
            {
                "mask": subsample_mask_labels_b,
                "cls": instance_cls_b,
                "box": instance_box_b,
            }
        )

    return subsample_inst_mask_arr

def get_subsample_gt_rays(
    instance_labels, subsample_idxs, instance_cls, instance_box, ray_labels, subsample_batch_offsets, batch_size, ignore_label=-100
):
    subsample_inst_mask_arr = []

    subsample_instance_labels = instance_labels[subsample_idxs]
    for b in range(batch_size):
        start, end = subsample_batch_offsets[b], subsample_batch_offsets[b + 1]
        n_subsample = end - start

        instance_labels_b = subsample_instance_labels[start:end]

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != ignore_label]

        unique_inst = unique_inst[instance_cls[unique_inst] >= 0]

        n_inst_gt = len(unique_inst)

        if n_inst_gt == 0:
            subsample_inst_mask_arr.append(None)
            continue

        instance_cls_b = instance_cls[unique_inst]
        instance_box_b = instance_box[unique_inst]
        ray_labels_b = ray_labels[unique_inst]

        subsample_mask_labels_b = torch.zeros(
            (n_inst_gt, n_subsample), device=subsample_instance_labels.device, dtype=torch.float
        )
        # breakpoint()
        for i, uni_id in enumerate(unique_inst):
            mask_ = instance_labels_b == uni_id
            subsample_mask_labels_b[i] = mask_.float()

        subsample_inst_mask_arr.append(
            {
                "mask": subsample_mask_labels_b,
                "cls": instance_cls_b,
                "box": instance_box_b,
                'rays': ray_labels_b
            }
        )

    return subsample_inst_mask_arr
def get_spp_gt_ray(
    instance_labels, spps, instance_cls, instance_box, ray_labels, batch_offsets, batch_size, ignore_label=-100, pool=True
):
   
    spp_inst_mask_arr = []
    for b in range(batch_size):
        start, end = batch_offsets[b], batch_offsets[b + 1]

        instance_labels_b = instance_labels[start:end]
        spp_b = spps[start:end]
        spp_b_unique, spp_b = torch.unique(spp_b, return_inverse=True)
       
       
        n_spp = len(spp_b_unique)
        
        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != ignore_label]
        

        unique_inst = unique_inst[instance_cls[unique_inst] >= 0]
        
        n_inst_gt = len(unique_inst)

        if n_inst_gt == 0:
            spp_inst_mask_arr.append(None)
            continue
        
        instance_cls_b = instance_cls[unique_inst]
        instance_box_b = instance_box[unique_inst]
        ray_labels_b = ray_labels[unique_inst]

        spp_mask_labels_b = torch.zeros((n_inst_gt, n_spp), device=instance_labels.device, dtype=torch.float)

        for i, uni_id in enumerate(unique_inst):
            mask_ = instance_labels_b == uni_id
            spp_mask_ = custom_scatter_mean(mask_, spp_b, pool=pool, output_type=torch.float32)
            
            cond_ = spp_mask_ >= 0.5 # Finding super-points that have more than half of its belonging voxels True as gt mask 
            spp_mask_labels_b[i] = cond_.float()

        spp_inst_mask_arr.append(
            {
                "mask": spp_mask_labels_b,
                "cls": instance_cls_b,
                "box": instance_box_b,
                "rays": ray_labels_b
            }
        )

    return spp_inst_mask_arr

def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)
def lineseg_dist_torch(point_p,aa,bb):
    ###
    #point_p_t, aa_t, bb_t = torch.tensor(point_p), torch.tensor(aa), torch.tensor(bb)
    #point_p_t, aa_t, bb_t = point_p_t[None,:].repeat(10,1), aa_t[None,:].repeat(10,1), bb_t[None,:].repeat(10,1)
    
    dd = (bb - aa) / torch.linalg.norm(bb-aa)
    ss_t_ein = torch.einsum('ab,ab->a',aa - point_p, dd )
    tt_t_ein = torch.einsum('ab,ab->a',point_p - bb, dd )
    #ss_t = torch.mm((aa_t - point_p_t).T, dd_t)
    #tt_t = torch.mm((point_p_t - bb_t).T, dd_t)

    compared = torch.cat([ss_t_ein[:,None],tt_t_ein[:,None],torch.zeros(len(ss_t_ein)).float().cuda()[:,None]],1)
    
    hh_t = torch.amax(compared,1)
    cc_t = torch.cross(point_p - aa, dd)
    dist_torch = torch.hypot(hh_t, torch.linalg.norm(cc_t))

    return dist_torch
def lineseg_dist(point_p, aa, bb):

    # normalized tangent vector
    
    
    ###

    dd = np.divide(bb - aa, np.linalg.norm(bb - aa))

    # signed parallel distance components
    
    #ss = np.tensordot(aa[None,:] - point_p[None,:], dd[None,:], axes=1)

    ss = np.dot(aa - point_p, dd)
    tt = np.dot(point_p - bb, dd)
    
    # clamped parallel distance
    hh = np.maximum.reduce([ss, tt, 0])

    # perpendicular distance component
    cc = np.cross(point_p - aa, dd)

    return np.hypot(hh, np.linalg.norm(cc))
def get_iou_with_conf(gt_mask, sub_mask, ori_ref, conf_val, coords=None, ray_mask=None, is_ray=False, ray_center=None):

    cond_ = sub_mask >= conf_val
    gt_mask_ori_size = cond_[ori_ref]
    if is_ray:
        ray_mask[ray_mask==-100] = 0
        ray_mask = ray_mask == 1
        false_negatives = (gt_mask_ori_size==False) * (gt_mask)
        false_negatives_ray = (ray_mask == False) * (gt_mask)
        gt_mask_ori_size = gt_mask_ori_size * ray_mask
        

        cond_fn = (sub_mask < conf_val) * (sub_mask > 0.01)

        centered_points = coords - ray_center[None,-3:,0]
        new_pts = cartesian_to_spherical(centered_points)
        dists = new_pts[:,3]
        if cond_fn.sum().item()>0:
            
            locs = torch.where(cond_fn)[0]
            for locs_idx in range(len(locs)):
                ori_points = ori_ref ==locs[locs_idx] 
                #ori_points = coords[ori_points]
                ori_dists = dists[ori_points]
                ori_label = gt_mask[ori_points]

                pos_max_dist = ori_dists[ori_label].max()

                ori_dists_mask = ori_dists <= pos_max_dist
                acc = (ori_label * ori_dists_mask).sum() / ori_dists_mask.shape[0]
                if acc.item() > 0.5:
                    cond_[locs[locs_idx]] = True

        gt_mask_ori_size = cond_[ori_ref] * ray_mask
        false_negatives_combined = (gt_mask_ori_size==False) * (gt_mask)
        '''
        print('FP, ori : {0}, ray : {1}, combined : {2}'.format(false_negatives.sum(),
                                                                false_negatives_ray.sum(),
                                                                false_negatives_combined.sum()))


        '''
        
    overlap = gt_mask * gt_mask_ori_size
    iou_upper = overlap.sum() / (gt_mask.sum() + gt_mask_ori_size.sum() - overlap.sum())
    
    return iou_upper.item(), cond_


