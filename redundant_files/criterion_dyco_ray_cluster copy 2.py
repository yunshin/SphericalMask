import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import (batch_giou_corres, batch_giou_corres_polar, giou_aabb, 
                          giou_aabb_polar, get_mask_from_polar_single, get_3D_locs_from_rays, 
                          cdf_sequential, get_points_mask_from_ray,
                          try_vectorize, try_vectorize_mask2ray)
import torchvision
import numpy as np
import pdb
import time
@torch.no_grad()
def get_iou(inputs, targets, thresh=0.5):
    inputs_bool = inputs.detach().sigmoid()
    inputs_bool = inputs_bool >= thresh

    intersection = (inputs_bool * targets).sum(-1)
    union = inputs_bool.sum(-1) + targets.sum(-1) - intersection

    iou = intersection / (union + 1e-6)

    return iou


def compute_dice_loss(inputs, targets, num_boxes, mask=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()

    if mask is not None:
        inputs = inputs * mask
        targets = targets * mask

    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_boxes + 1e-6)


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mask=None):
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
    if mask is not None:
        ce_loss = ce_loss * mask

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


class Criterion_Dyco_Ray_Cluster(nn.Module):
    def __init__(
        self,
        semantic_classes=20,
        instance_classes=18,
        semantic_weight=None,
        ignore_label=-100,
        eos_coef=0.1,
        semantic_only=True,
        total_epoch=40,
        trainall=False,
        voxel_scale=50,
    ):
        super(Criterion_Dyco_Ray_Cluster, self).__init__()

        self.matcher = HungarianMatcher()
        self.semantic_only = semantic_only

        self.ignore_label = ignore_label

        self.label_shift = semantic_classes - instance_classes
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.eos_coef = eos_coef
        self.voxel_scale = voxel_scale

        self.total_epoch = total_epoch

        self.trainall = trainall

        empty_weight = torch.ones(self.instance_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.semantic_weight:
            for i in range(self.label_shift, self.instance_classes + self.label_shift):
                empty_weight[i - self.label_shift] = self.semantic_weight[i]

        self.register_buffer("empty_weight", empty_weight)

        # self.loss_weight = {
        #     "dice_loss": 4,
        #     # "focal_loss": 4,
        #     "bce_loss": 4,
        #     "cls_loss": 1,
        #     "iou_loss": 1,
        #     "box_loss": 1,
        #     "giou_loss": 1,
        # }

        self.loss_weight = {
            "dice_loss": 1,
            "bce_loss": 1,
            "cls_loss": 0.5,
            "iou_loss": 0.5,
            "box_loss": 0.5,
            "giou_loss": 0.5,
            #'mask_ray_loss':1,
            #'angular_focal_loss':1,
            'mask_inside_loss':1,   
        }
        self.focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss
    def set_ray_head(self, func_):
        self.ray_head = func_
    def set_angles_ref(self, angles_ref, width, height):
        self.angles_ref = angles_ref
        self.rays_width = width
        self.rays_height = height
    def set_anchor(self, anchors):
        
        self.rays_anchors_tensor = anchors
        
        self.is_anchors = True
    def cal_point_wise_loss(
        self,
        semantic_scores,
        centroid_offset,
        corners_offset,
        box_conf,
        angular_offset,

        semantic_labels,
        instance_labels,
        centroid_offset_labels,
        corners_offset_labels,
        angular_offset_labels,
        coords_float,
    ):
        losses = {}

        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device=semantic_labels.device)
        else:
            weight = None
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, ignore_index=self.ignore_label, weight=weight
        )

        losses["pw_sem_loss"] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        total_pos_inds = pos_inds.sum()
        if total_pos_inds == 0:
            offset_loss = 0 * centroid_offset.sum()
            offset_vertices_loss = 0 * corners_offset.sum()
            conf_loss = 0 * box_conf.sum()
            giou_loss = 0 * box_conf.sum()
        else:
            offset_loss = (
                F.l1_loss(centroid_offset[pos_inds], centroid_offset_labels[pos_inds], reduction="sum")
                / total_pos_inds
            )
            
            offset_vertices_loss = (
                F.l1_loss(corners_offset[pos_inds], corners_offset_labels[pos_inds], reduction="sum") / total_pos_inds
            )
            
            if len(angular_offset_labels) > 1 :
               
                angular_offset_sim = F.cosine_similarity(angular_offset[pos_inds], angular_offset_labels[pos_inds])
                angular_offset_loss = 1 - angular_offset_sim
                angular_offset_loss = angular_offset_loss.sum() / total_pos_inds

                angular_offset_l1 = F.l1_loss(angular_offset[pos_inds], angular_offset_labels[pos_inds], reduction="sum")
                angular_offset_l1 = angular_offset_l1 / total_pos_inds

                angular_offset_loss += angular_offset_l1
                
            else:
                angular_offset_loss = torch.zeros(1).float().cuda()


            
            #pdb.set_trace()
            if len(corners_offset.shape) == 2:
                iou_gt, giou = batch_giou_corres(
                    corners_offset[pos_inds] + coords_float[pos_inds].repeat(1, 2),
                    corners_offset_labels[pos_inds] + coords_float[pos_inds].repeat(1, 2),
                )
            else:
                
                iou_gt, giou = batch_giou_corres_polar(
                    corners_offset[pos_inds] + coords_float[pos_inds][:,None,:].repeat(1,corners_offset.shape[1],1),
                    corners_offset_labels[pos_inds] + coords_float[pos_inds][:,None,:].repeat(1,corners_offset.shape[1],1),
                )
            
            giou_loss = torch.sum((1 - giou)) / total_pos_inds

            iou_gt = iou_gt.detach()
            conf_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction="sum") / total_pos_inds

        losses["pw_center_loss"] = offset_loss * self.voxel_scale / 50.0
        losses["pw_corners_loss"] = offset_vertices_loss * self.voxel_scale / 50.0
        losses["pw_angular_loss"] = angular_offset_loss
        losses["pw_giou_loss"] = giou_loss
        losses["pw_conf_loss"] = conf_loss

        return losses
    def get_anchors(self):
        return self.rays_anchors_tensor
    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        mask_ray_logits_list,
        conf_logits,
        box_preds,
        proto_coff,

        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        ray_labels,
        angular_labels,
        batch_size,
        dc_coords_float,
        dc_coords_float_mask,
        dc_batch_offset,
        dc_proto_features,
        is_mask=True,
        is_aux=False
    ):
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)

        num_gt = 0
        num_angular_samples_total = 0
        pos_angular_samples_total = 0

        num_mask_total = 0
        pos_mask_total = 0
       
        for b in range(batch_size):
            if is_mask:
                mask_logit_b = mask_logits_list[b]
                mask_logit_ray_b = mask_ray_logits_list[b]

                
            else:
                mask_logit_b = box_preds[b].clone().detach()
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]
            proto_coff_b = proto_coff[b]
            dc_coords_float_mask_b = dc_coords_float_mask[b]
            pred_inds, cls_label, inst_label, box_label, ray_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b], ray_labels[b]
            #angular_label_b = angular_labels[b]
          
            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue
            if is_mask:
                mask_logit_pred = mask_logit_b[pred_inds]
                mask_logit_ray_pred = mask_logit_ray_b[pred_inds]
                proto_coff_b = proto_coff_b[pred_inds]
                dc_coords_mask_b_ = dc_coords_float_mask_b[pred_inds]
            dc_coords_float_b = dc_coords_float[dc_batch_offset[b]:dc_batch_offset[b+1]]
            dc_proto_features_b = dc_proto_features[dc_batch_offset[b]:dc_batch_offset[b+1]]
            #dc_coords_mask_b_ = dc_coords_float_mask_b[dc_batch_offset[b]:dc_batch_offset[b+1]]

            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
            
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
            if is_mask:
                
                start_time_vec = time.time()
                #rays_label_anchors = self.rays_anchors_tensor[cls_label]
                #chk_max = ray_label[:,:-3] > rays_label_anchors
                if False and chk_max.sum().item() > 0:
                    print('values bigger than anchor found. Updating..')
                    locs = torch.where(chk_max)
                    cls_change, ray_num_change = locs[0], locs[1]
                    cls_label_change = cls_label[cls_change]
                    
                    print('{0},{1}, {2} values: {3}'.format(cls_change, ray_num_change, self.rays_anchors_tensor.shape,
                                                            ray_label[cls_change, ray_num_change]))
                    #pdb.set_trace()
                    self.rays_anchors_tensor[cls_label_change, ray_num_change] = ray_label[cls_change, ray_num_change]
                    
                    try:
                        np.save('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_max_vals_{0}_{1}.npy'.format(self.rays_height, self.rays_width), self.rays_anchors_tensor.cpu().numpy())
                    except:
                        pdb.set_trace()
                    #with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_max_vals_{0}_{1}.pkl'.format(rays_height, rays_width), 'wb') as f:
                    #    pickle.dump(self.anchors_max, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    #    self.anchors_max = pickle.load(f)
                #rays_label_anchors = self.rays_anchors_tensor[cls_label]    
                #rays_label_anchors_form = torch.cat([rays_label_anchors, ray_label[:,-3:]],1)
                
                #rays_label_anchors_form[:,:-3] -= (mask_logit_ray_pred[:,:-3])
                #rays_label_anchor - mask_logit_ray_pred
                
                #New
                '''
                if (mask_logit_pred>=0).sum().item() == 0:
                    
                    rays_from_mask = try_vectorize_mask2ray(dc_coords_float,  (mask_logit_pred>=0)*1.0,  self.rays_width, self.rays_height)
                    exist = True
                else:
                    exist = False
                    rays_from_mask = try_vectorize_mask2ray(dc_coords_float.clone(),  (inst_label>0)*1.0,  self.rays_width, self.rays_height)
                    #rays_from_mask = try_vectorize_mask2ray(dc_coords_float,  (mask_logit_pred>=0)*1.0,  self.rays_width, self.rays_height)
                
                '''
                with torch.no_grad():
                    #
                    #if exist:
                    #    dist_anchors_pred, angles_num_pred, roi_locs_pred = try_vectorize(dc_coords_float_b, inst_label, rays_from_mask, self.rays_width, self.rays_height)
                    #else:
                    dist_anchors, angles_num, roi_locs_pred = try_vectorize(dc_coords_float_b.clone(), inst_label, ray_label, self.rays_width, self.rays_height)
                    #dist_anchors_pred, angles_num_pred, roi_locs_pred = try_vectorize(dc_coords_float_b, inst_label, mask_logit_ray_pred, self.rays_width, self.rays_height)
                    #dist_anchors_pred, angles_num_pred, roi_locs_pred = try_vectorize(dc_coords_float_b, inst_label, rays_label_anchors_form, self.rays_width, self.rays_height)
                '''
                #Test2 to make roi extract same as old one
                all_masks_ray_, all_pred_mask_baseline, final_pred_mask, roi_indice = get_points_mask_from_ray(dc_coords_float_b, inst_label, ray_label, self.rays_width, self.rays_height,with_pred=True, pred_mask=mask_logit_pred)
                for chk_idx in range(len(dist_anchors)):
                    idx_indice = roi_locs_pred[1][roi_locs_pred[0] == chk_idx]
                    idx_indice_old = roi_indice[chk_idx]
                    if len(idx_indice) != len(idx_indice_old):
                        pdb.set_trace()
                    idx_indice_sorted,_ = torch.sort(idx_indice)
                    idx_indice_sorted_old,_ = torch.sort(idx_indice_old)
                   
                    mask = idx_indice_sorted_old == idx_indice_sorted
                    roi_locs_pred[1][roi_locs_pred[0] == chk_idx] = idx_indice_old
                    if mask.sum().item() != len(mask):
                        pdb.set_trace()
                mask = roi_locs_pred[1][roi_locs_pred[0] == 0] == roi_indice[0]
                if mask.sum().item() != len(mask):
                        pdb.set_trace()
                #Test 2 Ends
                '''
                '''
                if exist:
                    
                    for idx_chk in range(len(mask_logit_pred)):
                        mask_idx = mask_logit_pred[idx_chk] >= 0
                        mask_from_ray = torch.zeros_like(mask_idx)
                        
                        mask_from_ray[roi_locs_pred[1][roi_locs_pred[0]==idx_chk]] = 1
                        FN = (mask_from_ray == 0) * (mask_idx == 1) 
                        TP = (mask_from_ray == 1) * (mask_idx == 1) 
                        if TP.sum().item() < mask_idx.sum().item():
                            print('False')
                            pdb.set_trace()
                '''
                
                
             

                '''
                sel_idx = 0
                for chk_idx in range(len(roi_indice)):
                    num_manual = roi_indice[chk_idx]
                    num_vec = roi_locs[1][roi_locs[0] == chk_idx]
                    
                    if num_manual.shape[0] != num_vec.shape[0]:
                        print('mismatch')
                        pdb.set_trace()
                print('checking done')
                '''
                
                pred_inst_mask, gt_mask_ray_inst, ious, ious_baseline = [],[], [], []

                ##############this part?
                '''
                
                roi_pred_list = []
                roi_gt_list = []
                for mask_idx in range(len(mask_logit_pred)):
                    proto_feature = dc_proto_features_b[roi_locs_pred[1][roi_locs_pred[0] == mask_idx]].squeeze()
                    #proto_coff_b[mask_idx]
                    #if mask_idx == 0:
                    #    print(proto_coff_b[mask_idx])
                    
                    roi_pred = torch.matmul(F.relu(proto_feature), proto_coff_b[mask_idx][:,None])
                    roi_pred_list.append(roi_pred)
                    
                    roi_gt = inst_label[mask_idx][roi_locs_pred[1][roi_locs_pred[0]==mask_idx]]
                    roi_gt_list.append(roi_gt)
                
                roi_pred_total = torch.cat(roi_pred_list).squeeze()
                roi_gt_total = torch.cat(roi_gt_list)
                '''
                
                #New
                
                ###Trial with same head for mask
                #proto_coff_b[roi_locs_pred[0],roi_locs_pred[1]]
                ###
                #dc_coords_mask_b_[roi_locs_pred[0], roi_locs_pred[1]]
                #roi_pred_total = torch.matmul(F.relu(dc_proto_features_b[roi_locs_pred[1]].squeeze()), proto_coff_b.T)#ori
                '''
                try:
                    feat = dc_coords_mask_b_[roi_locs_pred[0], roi_locs_pred[1]].squeeze()
                except:
                    pdb.set_trace()
                feat2 = dc_proto_features_b[roi_locs_pred[1]].squeeze()
                feats = feat+feat2
                '''
                
                feats = dc_proto_features_b[roi_locs_pred[1]].squeeze()
                
                #roi_pred_total = torch.matmul(F.relu(dc_coords_mask_b_[roi_locs_pred[0], roi_locs_pred[1]].squeeze()), proto_coff_b.T)
                roi_pred_total = torch.matmul(F.relu(feats), proto_coff_b.T)
                roi_pred_total = roi_pred_total[torch.arange(len(roi_pred_total)).long().cuda(),roi_locs_pred[0]]
                roi_gt_total = inst_label[roi_locs_pred[0], roi_locs_pred[1]]
                
               
                dice_loss_ray = compute_dice_loss(roi_pred_total[None,:], roi_gt_total[None,:], 1)
                
                ray_bce_loss = F.binary_cross_entropy_with_logits(roi_pred_total, roi_gt_total, reduction="none")
                ray_bce_loss = ray_bce_loss.sum() / num_gt_batch
                #New Ends
                

                '''
                #roi_pred_total = torch.einsum('abc,cdf->ac', dc_proto_features_b[roi_locs_pred[1]],  proto_coff_b[roi_locs_pred[0]][None,:])
                
                #torch.where(inst_label[0]==1)

               
                roi_preds, roi_gts = [], []
                dice_loss_ray = torch.zeros(1).float().cuda()
                for mask_idx in range(len(inst_label)):
                    roi_indice_idx = roi_locs_pred[1][roi_locs_pred[0] == mask_idx]
                    roi_mask_pred = dc_proto_features_b[roi_indice_idx,:,0]
                    
                    roi_mask_pred = torch.matmul(dc_proto_features_b[roi_indice_idx].squeeze(),proto_coff_b[mask_idx][:,None])
                    
                    gt_mask_ray = inst_label[mask_idx, roi_indice_idx]
                    if roi_mask_pred.ndim ==2:
                        roi_mask_pred = roi_mask_pred.squeeze()
                    
                    intersection = ((roi_mask_pred.squeeze()>=0) * (gt_mask_ray == 1)).sum().item()
                    iou = intersection / ((roi_mask_pred>=0).sum().item() + (gt_mask_ray==1).sum().item() - intersection + 1e-6)
                    ious.append(iou)

                    
                    mask_logit_baseline = mask_logit_pred[mask_idx,roi_indice_idx]
                    intersection = ((mask_logit_baseline >=0) * (gt_mask_ray == 1)).sum().item()
                    iou_baseline = intersection / ((mask_logit_baseline>=0).sum().item() + (gt_mask_ray==1).sum().item() - intersection + 1e-6)
                    ious_baseline.append(iou_baseline)

                    roi_preds.append(roi_mask_pred)
                    roi_gts.append(gt_mask_ray)
                    
                    if len(gt_mask_ray) == 1:
                        
                        roi_mask_pred = roi_mask_pred.repeat(2)
                        gt_mask_ray = gt_mask_ray.repeat(2)
                    
                    dice_ = compute_dice_loss(roi_mask_pred[None,:], gt_mask_ray[None,:], 1)
                    dice_loss_ray += dice_
                try:
                    roi_preds = torch.cat(roi_preds)
                except:
                    pdb.set_trace()
                roi_gts = torch.cat(roi_gts)
                
                dice_loss_ray = dice_loss_ray/num_gt_batch
                #dice_loss_ray = compute_dice_loss(roi_preds.squeeze()[None,:], roi_gts[None,:], 1)
                ray_bce_loss = F.binary_cross_entropy_with_logits(roi_preds, roi_gts, reduction="none")
                ray_bce_loss = ray_bce_loss.sum() / num_gt_batch


                ious = np.array(ious)
                ious_baseline = np.array(ious_baseline)
                #print('iou roi: {0} iou baseline: {1}'.format(ious.mean(), ious_baseline.mean()))
                '''
                
                
                
                
                '''
                ##Old code that works
                all_masks_ray_, all_pred_mask_baseline, final_pred_mask, roi_indice = get_points_mask_from_ray(dc_coords_float_b, inst_label, ray_label, self.rays_width, self.rays_height,with_pred=True, pred_mask=mask_logit_pred)
                FP_ray, FN_ray, FP_chk, FN_chk = 0,0,0,0
                P_vals, N_vals = [],[]
                
                pred_inst_mask, gt_mask_ray_inst, ious, ious_baseline = [],[], [], []
                baseline_mask = []
                roi_preds, roi_gts = [], []
                dice_loss_ray = torch.zeros(1).float().cuda()
                '''
                
                ###Test1 
                '''
                gts = []
                for mask_idx in range(len(all_masks_ray_)):
                    gt_mask_ray = torch.cat(all_masks_ray_[mask_idx])
                    
                    indice_ = torch.zeros_like(gt_mask_ray)
                    indice_[:] = mask_idx
                    gt_mask_ray = torch.cat([gt_mask_ray[:,None], indice_[:,None]],1)
                    gts.append(gt_mask_ray)
                roi_gts = torch.cat(gts)
                roi_indice_all = torch.cat(roi_indice)

                
                
                base_features = dc_proto_features_b[roi_indice_all]
                
                roi_pred_total = torch.matmul(F.relu(base_features.squeeze()), proto_coff_b.T)
                roi_preds = roi_pred_total[torch.arange(len(roi_pred_total)).long().cuda(),roi_gts[:,1].long()]
                roi_gts = roi_gts[:,0]

                dice_loss_ray = compute_dice_loss(roi_preds.squeeze()[None,:], roi_gts[None,:], 1)
                
                
                #dice_loss_ray = compute_dice_loss(roi_pred_total.squeeze()[None,:], roi_gts[None,:], 1)
                #ray_bce_loss = F.binary_cross_entropy_with_logits(roi_pred_total, roi_gts, reduction="none")
                #ray_bce_loss = F.binary_cross_entropy_with_logits(roi_preds, roi_gts, reduction="none")
                #ray_bce_loss = ray_bce_loss.sum() / num_gt_batch
                ###Test1 Ends 
                '''
                
                '''
                #Old code that works # original one that should be commented out
                for mask_idx in range(len(all_masks_ray_)):
                    
                    gt_mask_ray = torch.cat(all_masks_ray_[mask_idx])
                    pred_mask_baseline = all_pred_mask_baseline[mask_idx]
                    roi_indice_idx = roi_indice[mask_idx]

                    
                    #roi_mask_pred = dc_proto_features_b[roi_indice_idx].squeeze() @ proto_coff_b[mask_idx][:,None]
                    roi_mask_pred = torch.matmul(dc_proto_features_b[roi_indice_idx].squeeze(),proto_coff_b[mask_idx][:,None])
                    if roi_mask_pred.ndim ==2:
                        roi_mask_pred = roi_mask_pred.squeeze()
                    #if b == 3 and mask_idx == 12:
                    #    pdb.set_trace()
                    #print('{0},{1} : {2}'.format(b,mask_idx, len(roi_mask_pred)))
                    if len(gt_mask_ray) == 0 :
                        continue
                    
                    roi_preds.append(roi_mask_pred)
                    roi_gts.append(gt_mask_ray)
                    if len(gt_mask_ray) == 1:
                        
                        roi_mask_pred = roi_mask_pred.repeat(2)
                        gt_mask_ray = gt_mask_ray.repeat(2)
                    
                    dice_ = compute_dice_loss(roi_mask_pred.squeeze()[None,:], gt_mask_ray[None,:], 1)
                    dice_loss_ray += dice_

                    intersection = ((roi_mask_pred>=0) * (gt_mask_ray == 1)).sum().item()
                    iou = intersection / ((roi_mask_pred>=0).sum().item() + (gt_mask_ray==1).sum().item() - intersection + 1e-6)
                    ious.append(iou)


                    mask_logit_baseline = mask_logit_pred[mask_idx,roi_indice_idx]
                    intersection = ((mask_logit_baseline >=0) * (gt_mask_ray == 1)).sum().item()
                    iou_baseline = intersection / ((mask_logit_baseline>=0).sum().item() + (gt_mask_ray==1).sum().item() - intersection + 1e-6)
                    ious_baseline.append(iou_baseline)
                try:
                    roi_preds = torch.cat(roi_preds)
                except:
                    pdb.set_trace()
                roi_gts = torch.cat(roi_gts)
                dice_loss_ray = dice_loss_ray/num_gt_batch
                ious = np.array(ious)
                ious_baseline = np.array(ious_baseline) 
                #print('iou roi: {0} iou baseline: {1}'.format(ious.mean(), ious_baseline.mean()))
                

    
                
                
                #dice_loss_ray = compute_dice_loss(roi_preds.squeeze()[None,:], roi_gts[None,:], 1)
                ray_bce_loss = F.binary_cross_entropy_with_logits(roi_preds, roi_gts, reduction="none")
                ray_bce_loss = ray_bce_loss.sum() / num_gt_batch
                
                
                #Old codes ends here
                '''
                #Common code starts here
                loss_dict['mask_inside_loss'] = loss_dict['mask_inside_loss'] + 0#+ ray_bce_loss + dice_loss_ray
                
                loss_dict["dice_loss"] = loss_dict["dice_loss"]+ compute_dice_loss(mask_logit_pred, inst_label, num_gt_batch)
                mask_logit_pred_chk = mask_logit_pred > 0.0
                true_mask = inst_label == 1
                pos_mask_total += mask_logit_pred_chk[true_mask].sum().item()
                num_mask_total += true_mask.sum().item()
               
            if True:
                start_time = time.time()
                angular_label = []
                
                ori_mask = (mask_logit_pred.detach().clone() > 0) *1.0
                FP = (ori_mask == 1) * (inst_label == 0)
                FN = (ori_mask == 0) * (inst_label == 1)
                
                gt_for_mask2 = torch.zeros_like(inst_label)
                gt_for_mask2[FP] = 1
                gt_for_mask2[FN] = 1
                '''
                focal_loss = F.binary_cross_entropy_with_logits(angular_pred, gt_for_mask2, reduction="none")
                focal_loss = focal_loss.mean(1).sum() / (gt_for_mask2.sum() + 1e-6)
                #if gt_for_mask2.sum().item() > 0:
                #    pdb.set_trace()
                #focal_loss = self.focal_loss(angular_pred, gt_for_mask2)
                #focal_loss = focal_loss.sum() / (gt_for_mask2.sum() + 1e-6)
                
                pos_pred_vals = angular_pred[gt_for_mask2 == 1]
                neg_pred_vals = angular_pred[gt_for_mask2 == 0]
                
                if len(pos_pred_vals) > 0:
                    pos_min = torch.amin(pos_pred_vals).item()
                    pos_mean = torch.mean(pos_pred_vals).item()
                    pos_max = torch.amax(pos_pred_vals).item()
                else:
                    pos_min = -1000
                    pos_mean = -1000
                    pos_max = -1000
                if pos_max > 0:
                    acc = (pos_pred_vals > 0).sum() / len(pos_pred_vals)
                    print('ACC : {0} neg max : {1}'.format(acc.item(),torch.amax(neg_pred_vals).item()))
                loss_dict['angular_focal_loss'] = loss_dict['angular_focal_loss'] + focal_loss
                '''
                
                '''
                print('pos min/mean: {0}/{1} neg max/mean: {2}/{3}'.format(pos_min, pos_mean,
                                                                            torch.amax(neg_pred_vals).item(),
                                                                            torch.mean(neg_pred_vals).item()))
                '''
                
                #focal_loss = focal_loss.sum() / (gt_for_mask2.sum() + 1e-6)
                #minus_target = torch.zeros(len(angular_pred[angular_label_b[:,:,0]==1])).float().cuda()
                #plus_target = torch.zeros(len(angular_pred[angular_label_b[:,:,0]==0])).float().cuda()

                #minus_target[:] = -10
                #plus_target[:] = 10
                '''
                pos_dist = angular_pred[inst_label==1]
                if (pos_dist<1.0).sum().item() > 0:
                    
                    pos_dist = pos_dist[pos_dist<1.0].mean()
                else:
                    pos_dist = torch.zeros(1).float().cuda()
                neg_dist = angular_pred[inst_label==0]
                if (neg_dist>=0.0).sum().item() > 0:
                    neg_dist = neg_dist[neg_dist>=0.0].mean()
                else:
                    neg_dist = torch.zeros(1).float().cuda()
                '''
                #neg_dist = -1*neg_dist
                #pos_dist = F.l1_loss(angular_pred[angular_label_b[:,:,0]==1], minus_target)
                #neg_dist = F.l1_loss(angular_pred[angular_label_b[:,:,0]==0], plus_target)
                #print('pos min: {0}, neg max: {1}'.format(angular_pred[angular_label_b[:,:,0]==1].min(), angular_pred[angular_label_b[:,:,0]==0].max()))
                

                
                '''
                bce_loss_angular = F.binary_cross_entropy_with_logits(angular_pred, inst_label, reduction="none")
                bce_loss_angular = bce_loss_angular.mean(1).sum() / (num_gt_batch + 1e-6)

                loss_dict["angular_loss_neg"] = loss_dict["angular_loss_neg"] + compute_dice_loss(
                    angular_pred, inst_label, num_gt_batch
                ) #+ neg_dist

                loss_dict["angular_loss_pos"] = loss_dict["angular_loss_pos"] + bce_loss_angular #+ (-1*pos_dist)
                #loss_dict["angular_loss_neg"] = loss_dict["angular_loss_neg"] + neg_dist
                
                ori_mask = (mask_logit_pred.detach().clone() > 0) *1.0
                FP = (ori_mask == 1) * (inst_label == 0)
                FN = (ori_mask == 0) * (inst_label == 1)
                
                gt_for_mask2 = torch.zeros_like(inst_label)
                gt_for_mask2[FP] = -1
                gt_for_mask2[FN] = 1
                

                
                FP_pred = mask_logit_pred[FP]
                FN_pred = mask_logit_pred[FN]
                aux_gt = torch.zeros(len(FP_pred)+len(FN_pred)).float().cuda()
                aux_gt[:len(FP_pred)] = 0
                aux_gt[len(FP_pred):] = 1
                
                False_pred = torch.cat([FP_pred, FN_pred])
                aux_loss = F.binary_cross_entropy_with_logits(False_pred, aux_gt, reduction="none")
                aux_loss = aux_loss.mean()
                #loss_dict["angular_loss_pos"] = loss_dict["angular_loss_pos"] + aux_loss
                
               
                
                loss_dict['aux_loss'] = loss_dict['aux_loss'] + aux_loss
                '''
            if is_mask:
                
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss
            
                
               
                

            if is_mask:
                gt_iou = get_iou(mask_logit_pred, inst_label)
                '''
                loss_dict["iou_loss"] = (
                    loss_dict["iou_loss"] + F.mse_loss(conf_logits_pred, gt_iou_ray, reduction="sum") / num_gt_batch
                )
                '''
                loss_dict["iou_loss"] = (
                    loss_dict["iou_loss"] + F.mse_loss(conf_logits_pred, gt_iou, reduction="sum") / num_gt_batch
                )
                
            target_classes = (
                torch.ones((n_queries), dtype=torch.int64, device=cls_logits.device) * self.instance_classes
            )

            target_classes[pred_inds] = cls_label

            loss_dict["cls_loss"] = loss_dict["cls_loss"] + F.cross_entropy(
                cls_logit_b,
                target_classes,
                self.empty_weight,
                reduction="mean",
            )


            '''
            ray_pred = []
            for idx_cls in range(len(cls_label)):
                ray_pred.append(mask_logit_ray_pred[idx_cls, cls_label[idx_cls]][None,:])
            ray_pred = torch.cat(ray_pred)
            '''
            #ray_loss = F.l1_loss(mask_logit_ray_pred[:,cls_label], ray_label, reduction="none")
            
            '''
            ray_target = rays_label_anchors - ray_label[:,:-3]
            ray_target = ray_target / 10.0
            if ray_target.min().item() < 0:
                pdb.set_trace()
            ray_target = torch.cat([ray_target, ray_label[:,-3:]],1)
            '''
            
            ray_pred = mask_logit_ray_pred
            #ray_loss = F.l1_loss(ray_pred, ray_label, reduction="mean")
            #ray_loss = F.l1_loss(ray_pred, ray_target, reduction="none")
           
            #ray_loss = F.mse_loss(ray_pred[b,cls_label], ray_label, reduction="none")
            #ray_loss = ray_loss.sum() / (num_gt_batch + 1e-6)
            #loss_dict["mask_ray_loss"] = loss_dict["mask_ray_loss"]+ 0#  + ray_loss

            loss_dict["box_loss"] = (
                loss_dict["box_loss"]
                + (self.voxel_scale / 50.0) * F.l1_loss(box_pred, box_label, reduction="sum") / num_gt_batch
            )
            
            
            
            if is_mask :
                iou_gt, giou = giou_aabb(box_pred, box_label, coords=None)
            else:
                pdb.set_trace()
                box_pred = box_pred.reshape(box_pred.shape[0], int(box_pred.shape[1]/3), 3)
                box_label = box_label.reshape(box_label.shape[0], int(box_label.shape[1]/3), 3)
                iou_gt, giou = batch_giou_corres_polar(
                    box_pred,
                     box_label,
                )
                loss_dict["iou_loss"] = (
                    loss_dict["iou_loss"] + F.mse_loss(conf_logits_pred, iou_gt, reduction="sum") / num_gt_batch
                )
            
            
            loss_dict["giou_loss"] = loss_dict["giou_loss"] + torch.sum(1 - giou) / num_gt_batch
        
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / batch_size
        '''
        for k in loss_dict.keys():
            
            if k == 'aux_loss':
                loss_dict[k] = loss_dict[k] * 1e-3
        '''        
    
        if False :
            #if is_aux:
            #    print('Aux Angular acc : {0}'.format(pos_angular_samples_total/num_angular_samples_total))
            #    print('Aux Mask acc : {0}'.format(pos_mask_total/num_mask_total))
            #else:
            print('Angular acc : {0}'.format(pos_angular_samples_total/num_angular_samples_total))
            print('Mask acc : {0}'.format(pos_mask_total/num_mask_total))
        
        return loss_dict

    def forward(self, batch_inputs, model_outputs, is_mask=True):
        loss_dict = {}

        semantic_labels = batch_inputs["semantic_labels"]
        instance_labels = batch_inputs["instance_labels"]

        if model_outputs is None:
            loss_dict["Placeholder"] = torch.tensor(
                0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float
            )

            return loss_dict

        if self.semantic_only or self.trainall:
            # '''semantic loss'''
            semantic_scores = model_outputs["semantic_scores"]
            centroid_offset = model_outputs["centroid_offset"]
            corners_offset = model_outputs["corners_offset"]
            box_conf = model_outputs["box_conf"]
            

            if 'angular_offset' in model_outputs:
                angular_offset = model_outputs['angular_offset']
                angular_offset_labels = batch_inputs['angular_offset_labels']
            else:
                angular_offset = torch.zeros(1).float().cuda()
                angular_offset_labels = torch.zeros(1).float().cuda()

            coords_float = batch_inputs["coords_float"]
            centroid_offset_labels = batch_inputs["centroid_offset_labels"]
            corners_offset_labels = batch_inputs["corners_offset_labels"]

            point_wise_loss = self.cal_point_wise_loss(
                semantic_scores,
                centroid_offset,
                corners_offset,
                box_conf,
                angular_offset,

                semantic_labels,
                instance_labels,
                centroid_offset_labels,
                corners_offset_labels,
                angular_offset_labels,
                coords_float,
            )

            loss_dict.update(point_wise_loss)

            if self.semantic_only:
                return loss_dict

        for k in loss_dict.keys():
            if "pw" in k:
                loss_dict[k] = loss_dict[k] * 0.25

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float)
            loss_dict["aux_" + k] = torch.tensor(
                0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float
            )
        
        """ Main loss """
        cls_logits = model_outputs["cls_logits"]
        if 'mask_logits' in model_outputs:
            mask_logits = model_outputs["mask_logits"]
            mask_ray_logits = model_outputs['mask_ray_logits']
        else:
            mask_logits = []
            mask_ray_logits = []
        conf_logits = model_outputs["conf_logits"]
        box_preds = model_outputs["box_preds"]#Box pred includes anchor positions
        proto_coff = model_outputs['proto_coff']

        dc_coords_float = model_outputs['dc_coords_float']
        dc_coords_float_mask_feat = model_outputs['dc_coords_float_mask']
        dc_batch_offset = model_outputs['dc_batch_offsets']
        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]
        dc_mask_features_proto = model_outputs['dc_mask_features_angular']
        
        batch_size, n_queries = cls_logits.shape[:2]
        
        gt_dict, aux_gt_dict, matching_cost = self.matcher.forward_dup(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_inst_mask_arr,
            mask_ray_preds=mask_ray_logits,
            ray_labels=batch_inputs['rays_label'],
            dup_gt=4,
        )
        #print('matching cost : {0}'.format(matching_cost))
        # NOTE main loss

        row_indices = gt_dict["row_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]
        ray_labels = gt_dict['ray_labels']
        angular_labels = gt_dict['angular_labels']
        
        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            mask_ray_logits,
            conf_logits,
            box_preds,
            proto_coff,
            row_indices,

            cls_labels,
            inst_labels,
            box_labels,
            ray_labels,
            angular_labels,
            batch_size,
            dc_coords_float,
            dc_coords_float_mask_feat,
            dc_batch_offset,
            dc_mask_features_proto,
            is_mask=is_mask,
            is_aux=False
        )

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v

        # NOTE aux loss

        aux_row_indices = aux_gt_dict["row_indices"]
        aux_inst_labels = aux_gt_dict["inst_labels"]
        aux_cls_labels = aux_gt_dict["cls_labels"]
        aux_box_labels = aux_gt_dict["box_labels"]
        aux_ray_labels = aux_gt_dict['ray_labels']
        aux_angular_labels = aux_gt_dict['angular_labels']


        aux_main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            mask_ray_logits,
            conf_logits,
            box_preds,
            proto_coff,

            aux_row_indices,
            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            aux_ray_labels,
            aux_angular_labels,
            batch_size,
            dc_coords_float,
            dc_coords_float_mask_feat,
            dc_batch_offset,
            dc_mask_features_proto,
            is_mask=is_mask,
            is_aux=True
        )

        coef_aux = 2.0
        
        for k, v in self.loss_weight.items():
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux

        return loss_dict
