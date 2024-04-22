import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import (batch_giou_corres, batch_giou_corres_polar, giou_aabb, 
                          giou_aabb_polar, get_mask_from_polar_single, get_3D_locs_from_rays, 
                          cdf_sequential, get_points_mask_from_ray,
                          try_vectorize, try_vectorize_mask2ray)
import pdb
import torchvision
import numpy as np
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


class Criterion_Dyco_Ray_Cluster_Gamma(nn.Module):
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
        super(Criterion_Dyco_Ray_Cluster_Gamma, self).__init__()

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
            'second_bce_loss': 1,
            'second_dice_loss': 1,
            'second_l1_loss': 1,
        }
        self.train_iter = 0
    def set_train_iter(self, train_iter):
        self.train_iter = train_iter
    def set_mask_head(self, func):
        self.mask_head = func
    def get_loss_new(self, dc_coords_float_b, inst_label, ray_label, mask_feat_b, dc_proto_features_b, proto_coff_b, mask_logit_pred, ray_pred ):
        '''
        ###debug
        rays_from_mask = try_vectorize_mask2ray(dc_coords_float_b.clone(),  (inst_label>0)*1.0,  self.rays_width, self.rays_height, self.angles_ref)    
        dist_anchors_pred, angles_num_pred, roi_locs_pred = try_vectorize(dc_coords_float_b.clone(), inst_label, rays_from_mask, self.rays_width, self.rays_height)
        for debug_idx in range(len(rays_from_mask)):
            pdb.set_trace()
            gt_mask = inst_label[debug_idx]
            est_fore_locs = roi_locs_pred[1][roi_locs_pred[0] == debug_idx]
        pdb.set_trace()
        ##debug ends
        '''
        
        if False and (mask_logit_pred>=0).sum(1).min().item() == 0:
                with torch.no_grad():
                    rays_from_mask = try_vectorize_mask2ray(dc_coords_float_b.clone(),  (inst_label>0)*1.0,  self.rays_width, self.rays_height, self.angles_ref)    
                #dist_anchors_pred, angles_num_pred, roi_locs_pred = try_vectorize(dc_coords_float_b.clone(), inst_label, rays_from_mask, self.rays_width, self.rays_height)
                exist = False
        else:
            exist = True
           
            with torch.no_grad():
                #print('in')
                
                all_mask = ((inst_label>0)*1.0) + ((mask_logit_pred>=0)*1.0)
                all_mask = all_mask>=1
                rays_from_mask_pred = try_vectorize_mask2ray(dc_coords_float_b.clone(),  (mask_logit_pred>=0)*1.0,  self.rays_width, self.rays_height, self.angles_ref)    
                rays_from_mask = try_vectorize_mask2ray(dc_coords_float_b.clone(),  all_mask,  self.rays_width, self.rays_height, self.angles_ref)
                #rays_gt = try_vectorize_mask2ray(dc_coords_float_b.clone(),  inst_label,  self.rays_width, self.rays_height, self.angles_ref) 
                
                

                #rays_gt_from_another_centers = try_vectorize_mask2ray(dc_coords_float_b.clone(),  inst_label,  self.rays_width, self.rays_height, self.angles_ref,
                #                                                      center_provided=True,centers_=rays_from_mask_pred[:,-3:]) 
                
                #diff = rays_gt_from_another_centers[:,:-3] - rays_from_mask_pred[:,:-3]
                
        with torch.no_grad():
            #if exist:
            
            dist_anchors_pred, angles_num_pred_, roi_locs_pred, centerd_coords_ = try_vectorize(dc_coords_float_b.clone(), inst_label, rays_from_mask, self.rays_width, self.rays_height)
            

            #dist_anchors_pred_, angles_num_pred_, roi_locs_pred_ray, centerd_coords_ = try_vectorize(dc_coords_float_b.clone(), inst_label, ray_pred, self.rays_width, self.rays_height)
            
        '''
        ##baseline
        with torch.no_grad():
            
            FP_indice = (mask_logit_pred>=0) * (inst_label == 0)
            FN_indice = (mask_logit_pred<0) * (inst_label == 1)
        print('baseline - FP: {0} FN: {1} '.format(FP_indice.sum().item(), FN_indice.sum().item()))
        '''
        
        inverse_mask = -1*mask_logit_pred #for gamma clustering
        batch_num_array = torch.arange(len(angles_num_pred_))[:,None].repeat(1, angles_num_pred_.shape[1]).to(angles_num_pred_.device)
        angles_num_with_batch = torch.cat([batch_num_array[:,:,None], angles_num_pred_[:,:,None]],2) 
        angles_num_with_batch = angles_num_with_batch.reshape(-1,2)
        inst_label_flatten = inst_label.reshape(-1)
        centerd_coords_dist = centerd_coords_[:,:,3]

        
        #centerd_coords_dist = centerd_coords_dist[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]### Error
        #solution
        centerd_coords_dist = centerd_coords_dist.reshape(-1)
        ray_dist_with_angles = ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]

        with torch.no_grad():
            
            #FP_indice = (centerd_coords_dist + inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]] <= ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]])
            FP_indice = (centerd_coords_dist + inverse_mask.reshape(-1) <= ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]])
            FP_indice = FP_indice * (inst_label_flatten == 0)

            #FN_indice = (centerd_coords_dist + inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]) > ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
            FN_indice = (centerd_coords_dist + inverse_mask.reshape(-1)) > ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
            FN_indice = FN_indice * (inst_label_flatten == 1)
        
        epsilon = 0.6
        #FP_loss = 
        
        pred_target_gamma = inverse_mask.detach().clone()

        
        FP_ray_pred = ray_dist_with_angles[FP_indice]
        FN_ray_pred = ray_dist_with_angles[FN_indice]
       
        #err in inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
        #FP_cluster_pred = centerd_coords_dist + inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]

       

        #FP_pred_raw = -1*(inverse_mask.reshape(-1)[FP_indice])
        FP_cluster_pred = centerd_coords_dist + inverse_mask.reshape(-1)
        FP_cluster_pred = FP_cluster_pred[FP_indice]

        ##err in inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
        #FN_cluster_pred = centerd_coords_dist + inverse_mask[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
        
        #FN_pred_raw = -1*(inverse_mask.reshape(-1)[FN_indice])
        FN_cluster_pred = centerd_coords_dist + inverse_mask.reshape(-1)
        FN_cluster_pred = FN_cluster_pred[FN_indice]

        FP_sample_target_raw = -1*(FP_ray_pred - centerd_coords_dist[FP_indice])
        FN_sample_target_raw = -1*(FN_ray_pred - centerd_coords_dist[FN_indice])
       
        if len(FP_sample_target_raw) == 0:
            max_for_zero = torch.zeros(1).float().cuda()
            max_for_zero[0] = -1
            max_for_zero = max_for_zero.squeeze()
        else:
            max_for_zero = FP_sample_target_raw.max().detach()
        if len(FN_sample_target_raw) == 0:
            min_for_one = torch.zeros(1).float().cuda()
            min_for_one[0] = 1
            min_for_one = min_for_one.squeeze()
        else:
            min_for_one = FN_sample_target_raw.min().detach()

        needs_for_zero = max_for_zero + 1
        needs_for_one = min_for_one -1 
        epsilon_candi = torch.stack([torch.abs(needs_for_one),needs_for_zero])
        epsilon = torch.amax(epsilon_candi)
        epsilon = epsilon * 1.5


        FP_sample_target = FP_ray_pred + epsilon
        FN_sample_target = FN_ray_pred - epsilon

        FP_sample_target_raw2 = -1*(FP_sample_target - centerd_coords_dist[FP_indice])
        FN_sample_target_raw2 = -1*(FN_sample_target - centerd_coords_dist[FN_indice])

        
        if len(FP_sample_target) > 0 and FP_sample_target_raw2.max().item() > -0.98: 
            pdb.set_trace()
        if len(FN_sample_target) > 0 and FN_sample_target_raw2.min().item() < 0.98:
            pdb.set_trace()
        
        '''
        pred_target_gamma = pred_target_gamma.reshape(-1)
        pred_target_gamma[FP_indice] = FP_sample_target
        pred_target_gamma[FN_indice] = FN_sample_target
        pred_target_gamma = pred_target_gamma.reshape(inverse_mask.shape)
        
        inverse_mask_with_coords = centerd_coords_dist.reshape(inverse_mask.shape) + inverse_mask
        
        #ray_pred_loss = F.mse_loss(inverse_mask, pred_target_gamma, reduction='none')
        ray_pred_loss = torch.abs(inverse_mask_with_coords - pred_target_gamma)
        #ray_pred_loss = F.binary_cross_entropy_with_logits(inverse_mask, pred_target_gamma, reduction='none')

        ray_pred_loss = ray_pred_loss.sum() / (FP_indice.sum()+FN_indice.sum()+1e-6)
        print('gamma - FP: {0} FN: {1} loss: {2}'.format(len(FP_sample_target), len(FN_sample_target), ray_pred_loss.item()))
        '''
        
        if len(FP_sample_target) > 0:
            ray_FP_loss = torch.abs(FP_cluster_pred - FP_sample_target).mean()
            #ray_FP_loss = F.binary_cross_entropy_with_logits(FP_cluster_pred, FP_sample_target)
        else:
            ray_FP_loss = torch.zeros(1).float().cuda()
        if len(FN_sample_target) > 0:
           
            #ray_FN_loss = F.binary_cross_entropy_with_logits(FN_cluster_pred, FN_sample_target)
            ray_FN_loss = torch.abs(FN_cluster_pred - FN_sample_target).mean()
        else:
            ray_FN_loss = torch.zeros(1).float().cuda()
        
        

        ray_pred_loss = ray_FP_loss + ray_FN_loss
        #print('gamma - FP: {0} FN: {1} loss: {2}'.format(len(FP_sample_target), len(FN_sample_target), ray_pred_loss.item()))
        
        
        feats1 = mask_feat_b[roi_locs_pred[0],roi_locs_pred[1]]
        feats2 = dc_proto_features_b[roi_locs_pred[1]].squeeze()
        feats = feats1 + feats2
        
        feats = self.mask_head(F.relu(feats))
        roi_pred_total = torch.matmul(feats, proto_coff_b.T)
        roi_pred_total = roi_pred_total[torch.arange(len(roi_pred_total)).long().cuda(),roi_locs_pred[0]]
        #roi_pred_total = roi_pred_total[torch.arange(len(roi_pred_total)).long().cuda(),roi_locs_pred_refined[0]]
        roi_gt_total = inst_label[roi_locs_pred[0], roi_locs_pred[1]]
        #roi_gt_total = inst_label[roi_locs_pred_refined[0], roi_locs_pred_refined[1]]

        dice_loss_ray = compute_dice_loss(roi_pred_total[None,:], roi_gt_total[None,:], 1)
        #ray_bce_loss = compute_sigmoid_focal_loss(roi_pred_total[None,:], roi_gt_total[None,:], 1)
        ray_bce_loss = F.binary_cross_entropy_with_logits(roi_pred_total, roi_gt_total, reduction="none")
        
        pred_mask = roi_pred_total >= 0
        acc = (pred_mask == roi_gt_total).sum() / len(roi_gt_total)
        #print(acc.item())
        ray_bce_loss = ray_bce_loss.sum() / roi_gt_total.sum()
        
        ray_pred_loss = ray_pred_loss + F.l1_loss(ray_pred, ray_label)
        #ray_pred_loss = F.l1_loss(ray_pred, ray_label)
        #ray_pred_loss = ray_pred_loss + F.l1_loss(ray_pred, diff)
        
        '''
        ##debug
        with torch.no_grad():
            inclusion_ratio_pred_list, inclusion_ratio_to_gt_list, acc_ours_list, acc_baseline_list = [], [], [], []
            valid_point_inclusion_baseline_list = []
            valid_point_inclusion_list = []
            for idx_mask in range(len(inst_label)):
                inst_locs = torch.where(inst_label[idx_mask])[0]
                pred_locs = roi_locs_pred[1][roi_locs_pred[0]==idx_mask]
                pred_locs_all = roi_locs_pred_debug[1][roi_locs_pred_debug[0]==idx_mask]
                pred_locs_refined = roi_locs_pred_refined[1][roi_locs_pred_refined[0]==idx_mask]


                gt_mask = torch.zeros(len(inst_label[idx_mask]))
                
                gt_mask[inst_locs] = 1

                pred_mask = torch.zeros(len(inst_label[idx_mask]))
                pred_mask[pred_locs] = 1

                pred_mask_debug = torch.zeros(len(inst_label[idx_mask]))
                pred_mask_debug[pred_locs_all] = 1

                pred_mask_refined = torch.zeros(len(inst_label[idx_mask]))
                pred_mask_refined[pred_locs_refined] = 1


                included = ((gt_mask == 1) * (pred_mask == 1)).sum()   
            
                
                inclusion_ratio_to_gt = included / gt_mask.sum()
                inclusion_ratio_pred = included / pred_mask.sum()
                
                inclusion_ratio_valid_points = pred_mask_debug[inst_locs].sum() / len(inst_locs)
                inclusion_ratio_valid_points_gt = pred_mask[inst_locs].sum() / len(inst_locs)
                inclusion_ratio_valid_points_refined = pred_mask_refined[inst_locs].sum() / len(inst_locs)

                valid_point_inclusion_baseline_list.append(inclusion_ratio_valid_points)
                #valid_point_inclusion_list.append(inclusion_ratio_valid_points_gt)
                valid_point_inclusion_list.append(inclusion_ratio_valid_points_refined)
                mask_ray = inst_label[idx_mask][roi_locs_pred[1][roi_locs_pred[0]==idx_mask]]
                
                pred_vals = (roi_pred_total[roi_locs_pred[0] == idx_mask] >=0) * 1.0

                acc_new = (mask_ray == pred_vals).sum() / (len(mask_ray) + 1e-6)
                baseline_pred = (mask_logit_pred[idx_mask][roi_locs_pred[1][roi_locs_pred[0]==idx_mask]] >=0)*1.0
                acc_baseline = (mask_ray == baseline_pred).sum() / (len(mask_ray) + 1e-6)

                inclusion_ratio_pred_list.append(inclusion_ratio_pred.item())
                inclusion_ratio_to_gt_list.append(inclusion_ratio_to_gt.item())
                acc_ours_list.append(acc_new.item())
                acc_baseline_list.append(acc_baseline.item())

            inclusion_ratio_pred_list = np.array(inclusion_ratio_pred_list)
            inclusion_ratio_to_gt_list = np.array(inclusion_ratio_to_gt_list)
            acc_ours_list = np.array(acc_ours_list)
            acc_baseline_list = np.array(acc_baseline_list)
            valid_point_inclusion_list = np.array(valid_point_inclusion_list)
            valid_point_inclusion_baseline_list = np.array(valid_point_inclusion_baseline_list)
        print('valid points inclusion ratio inside mask(baseline/w ray): {0}/{1} valid points ratio inside pred mask: {2} acc baseline/ours: {3}/{4}'.format(
                                                                                                            round(np.mean(valid_point_inclusion_baseline_list),3),
                                                                                                            round(np.mean(valid_point_inclusion_list),3),
                                                                                                           round(np.mean(inclusion_ratio_pred_list),3),
                                                                                                           round(np.mean(acc_baseline_list),3),
                                                                                                           round(np.mean(acc_ours_list),3)))
        '''
        return dice_loss_ray, ray_bce_loss, ray_pred_loss
    def get_loss_old(self, dc_coords_float_b, inst_label, ray_label, mask_logit_pred, dc_proto_features_b, proto_coff_b, num_gt_batch):
        all_masks_ray_, all_pred_mask_baseline, final_pred_mask, roi_indice = get_points_mask_from_ray(dc_coords_float_b, inst_label, ray_label, self.rays_width, self.rays_height,with_pred=True, pred_mask=mask_logit_pred)
        roi_preds, roi_gts = [],[]
        ious, ious_baseline =[] ,[]
        dice_loss_ray = torch.zeros(1).float().cuda()
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
        ray_bce_loss = F.binary_cross_entropy_with_logits(roi_preds, roi_gts, reduction="none")
        ray_bce_loss = ray_bce_loss.sum() / num_gt_batch

        return dice_loss_ray, ray_bce_loss
    def set_angles_ref(self, angles_ref, width, height):
        self.angles_ref = angles_ref
        self.rays_width = width
        self.rays_height = height
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

    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        conf_logits,
        box_preds,
        ray_preds,
        proto_coff,
        mask_feats,
        row_indices,

        cls_labels,
        inst_labels,
        box_labels,
        ray_labels,
        batch_size,
        dc_coords_float,
        dc_batch_offset,
        dc_proto_features,
        is_mask=True
    ):
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)

        num_gt = 0
        for b in range(batch_size):
            if is_mask:
                mask_logit_b = mask_logits_list[b]
            else:
                mask_logit_b = box_preds[b].clone().detach()
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]
            ray_preds_b = ray_preds[b]
            proto_coff_b = proto_coff[b]
            mask_feat_b = mask_feats[b]
          
            pred_inds, cls_label, inst_label, box_label, ray_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b], ray_labels[b]
            dc_coords_float_b = dc_coords_float[dc_batch_offset[b]:dc_batch_offset[b+1]]
            dc_proto_features_b = dc_proto_features[dc_batch_offset[b]:dc_batch_offset[b+1]]


            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue
            if is_mask:
                mask_logit_pred = mask_logit_b[pred_inds]

            mask_feat_b = mask_feat_b.permute(0,2,1)
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
            ray_pred = ray_preds_b[pred_inds]
            proto_coff_b = proto_coff_b[pred_inds]
            mask_feat_b = mask_feat_b[pred_inds]
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
            
            ray_dice_loss, ray_bce_loss, ray_loss = self.get_loss_new(dc_coords_float_b, inst_label, ray_label, mask_feat_b, dc_proto_features_b, proto_coff_b, mask_logit_pred, ray_pred)
            #ray_dice_loss, ray_bce_loss = self.get_loss_old(dc_coords_float_b, inst_label, ray_label, mask_logit_pred, dc_proto_features_b, proto_coff_b, num_gt_batch)
            
        
            loss_dict['second_bce_loss'] = loss_dict['second_bce_loss'] + ray_bce_loss
            loss_dict['second_dice_loss'] = loss_dict['second_dice_loss'] + ray_dice_loss
            loss_dict['second_l1_loss'] = loss_dict['second_l1_loss'] + ray_loss
            if is_mask:
                loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(mask_logit_pred, inst_label, num_gt_batch)
            if is_mask:
                
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss
            
            if is_mask:
                gt_iou = get_iou(mask_logit_pred, inst_label)

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

            loss_dict["box_loss"] = (
                loss_dict["box_loss"]
                + (self.voxel_scale / 50.0) * F.l1_loss(box_pred, box_label, reduction="sum") / num_gt_batch
            )
            
            
            
            if is_mask :
                iou_gt, giou = giou_aabb(box_pred, box_label, coords=None)
            else:
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
        else:
            mask_logits = []
        conf_logits = model_outputs["conf_logits"]
        box_preds = model_outputs["box_preds"]#Box pred includes anchor positions
        proto_coff = model_outputs['proto_coff']
        mask_feats = model_outputs['mask_feats']
        ray_preds = model_outputs['ray_preds']

        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]
        dc_mask_features_proto = model_outputs['dc_mask_features_angular']
        dc_coords_float = model_outputs['dc_coords_float']
        dc_batch_offset = model_outputs['dc_batch_offsets']
       

        batch_size, n_queries = cls_logits.shape[:2]

        gt_dict, aux_gt_dict, matching_cost = self.matcher.forward_dup(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_inst_mask_arr,
            dup_gt=4,
        )
        
        # NOTE main loss

        row_indices = gt_dict["row_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]
        ray_labels = gt_dict['ray_labels']
        
        
        
        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            ray_preds,
            proto_coff,
            mask_feats,
            row_indices,
            

            cls_labels,
            inst_labels,
            box_labels,
            ray_labels,
            batch_size,
            dc_coords_float,
            dc_batch_offset,
            dc_mask_features_proto,
            is_mask=is_mask
        )

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v

        # NOTE aux loss

        aux_row_indices = aux_gt_dict["row_indices"]
        aux_inst_labels = aux_gt_dict["inst_labels"]
        aux_cls_labels = aux_gt_dict["cls_labels"]
        aux_box_labels = aux_gt_dict["box_labels"]
        aux_ray_labels = aux_gt_dict["ray_labels"]

        aux_main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            ray_preds,
            proto_coff,
            mask_feats,
            aux_row_indices,


            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            aux_ray_labels,
            batch_size,
            dc_coords_float,
            dc_batch_offset,
            dc_mask_features_proto,
            is_mask=is_mask
        )

        coef_aux = 2.0
        
        for k, v in self.loss_weight.items():
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux

        return loss_dict
