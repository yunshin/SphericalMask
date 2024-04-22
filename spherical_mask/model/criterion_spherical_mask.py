import torch
import torch.nn as nn
import torch.nn.functional as F

from spherical_mask.model.matcher import HungarianMatcher
from .model_utils import (batch_giou_corres, batch_giou_corres_polar, giou_aabb, find_sector, mask2ray)
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


class Criterion_SphericalMask(nn.Module):
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
        super(Criterion_SphericalMask, self).__init__()

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
    def set_mask_head(self, func, norm_func):
        self.mask_head = func
        self.norm = norm_func
    def set_activation(self, activation):
        self.activation = activation
    def get_loss(self, dc_coords_float_b, inst_label, ray_label, mask_feat_b, dc_proto_features_b, proto_coff_b, rpm_output, ray_pred ):
      
        
        
        with torch.no_grad():
        
            
            all_mask = ((inst_label>0)*1.0) + ((rpm_output>=0)*1.0)
            all_mask = all_mask>=1
            
            rays_from_mask = mask2ray(dc_coords_float_b.clone(),  all_mask,  self.rays_width, self.rays_height, self.angles_ref)
            rays_from_mask[:,:-3] = rays_from_mask[:,:-3] + ray_pred[:,:-3] 
            dist_anchors_pred, angles_num_pred_, roi_locs_pred, centerd_coords_ = find_sector(dc_coords_float_b.clone(), rpm_output, rays_from_mask, self.rays_width, self.rays_height, point_migration=True)  
            
        inverse_mask = -1*torch.tanh(rpm_output)

        batch_num_array = torch.arange(len(angles_num_pred_))[:,None].repeat(1, angles_num_pred_.shape[1]).to(angles_num_pred_.device)
        angles_num_with_batch = torch.cat([batch_num_array[:,:,None], angles_num_pred_[:,:,None]],2) 
        angles_num_with_batch = angles_num_with_batch.reshape(-1,2)
        inst_label_flatten = inst_label.reshape(-1)
        centerd_coords_dist = centerd_coords_[:,:,3]

        
     
        centerd_coords_dist = centerd_coords_dist.reshape(-1)
        ray_dist_with_angles = ray_pred[angles_num_with_batch[:,0],angles_num_with_batch[:,1]]
        
        with torch.no_grad():
          
            FP_indice = (rpm_output.reshape(-1) >= 0) * (inst_label_flatten == 0)
            FN_indice = (rpm_output.reshape(-1) < 0) * (inst_label_flatten == 1)
            TP_indice = (rpm_output.reshape(-1) >= 0) * (inst_label_flatten == 1)
            
        FP_ray_pred = ray_dist_with_angles[FP_indice]
        FN_ray_pred = ray_dist_with_angles[FN_indice]
       
        FP_cluster_pred = centerd_coords_dist + inverse_mask.reshape(-1)
        FP_cluster_pred = FP_cluster_pred[FP_indice]

        FN_cluster_pred = centerd_coords_dist + inverse_mask.reshape(-1)
        FN_cluster_pred = FN_cluster_pred[FN_indice]
        
        FP_sample_target_raw = FP_ray_pred - (centerd_coords_dist[FP_indice]+inverse_mask.reshape(-1)[FP_indice])
        FN_sample_target_raw = FN_ray_pred - (centerd_coords_dist[FN_indice]+inverse_mask.reshape(-1)[FN_indice])
       
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

        
        

        FP_sample_target = FP_ray_pred + 0.1
        FN_sample_target = FN_ray_pred - 0.1

       
        clustered_output_FP = centerd_coords_dist[FP_indice] - rpm_output.reshape(-1)[FP_indice]
        clustered_output_FN = centerd_coords_dist[FN_indice] - rpm_output.reshape(-1)[FN_indice]
        clustered_output_TP = centerd_coords_dist[TP_indice] - rpm_output.reshape(-1)[TP_indice]
        
        

        init = False
        if len(FP_sample_target) > 0:
            
            FP_input = (FP_ray_pred.detach() - clustered_output_FP)
            FP_target = torch.zeros(len(FP_input)).float().cuda()
            FP_target[:] = -1
        
            if init == False:
                inputs = FP_input
                targets = FP_target
                init=True
        
        if TP_indice.sum().item() > 0:
            TP_input = -1*clustered_output_TP
            TP_target = torch.zeros(len(TP_input)).float().cuda()
            TP_target[:] = 1
            

            if init :
                inputs = torch.cat([inputs, TP_input])
                targets = torch.cat([targets, TP_target])
            else:
                inputs = TP_input
                targets = TP_target
                init = True
            
            if len(FN_sample_target) > 0:
           
               
                FN_input = (FN_ray_pred.detach() - clustered_output_FN)
                
                FN_target = torch.zeros(len(FN_input)).float().cuda()
                FN_target[:] = 1
                if init :
                    inputs = torch.cat([inputs, FN_input])
                    targets = torch.cat([targets, FN_target])
                else:
                    inputs = FN_input
                    targets = FN_target
                    init = True
           
            if init:
                
                ray_pred_loss = F.soft_margin_loss(torch.tanh(inputs), targets)
            else:
                ray_pred_loss = torch.zeros(1).float().cuda()
        
        else:
            ray_pred_loss = torch.zeros(1).float().cuda()
        
        
        dice_loss_ray = torch.zeros(1).float().cuda()
        ray_bce_loss = torch.zeros(1).float().cuda()
        
        ray_pred_loss = ray_pred_loss + F.l1_loss(ray_pred, ray_label)
        
        return dice_loss_ray, ray_bce_loss, ray_pred_loss
    

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
            
            ray_dice_loss, ray_bce_loss, ray_loss = self.get_loss(dc_coords_float_b, inst_label, ray_label, mask_feat_b, dc_proto_features_b, proto_coff_b, mask_logit_pred, ray_pred)
            loss_dict['second_bce_loss'] = loss_dict['second_bce_loss'] + ray_bce_loss
            loss_dict['second_dice_loss'] = loss_dict['second_dice_loss'] + ray_dice_loss
            loss_dict['second_l1_loss'] = loss_dict['second_l1_loss'] + ray_loss
            if is_mask:
                #if 'gamma_clustering' in self.activation :
                loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(mask_logit_pred, inst_label, num_gt_batch)
            if is_mask:
                
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                
                #if 'gamma_clustering' in self.activation :
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
                loss_dict[k] = loss_dict[k] * 0.25 #ori code
                #loss_dict[k] = loss_dict[k] * 0.0000001 #ori code
                #loss_dict[k] = loss_dict[k] 

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
            #dup_gt=4,
            dup_gt=4
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
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v #ori
            #loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v * 0.25
            #loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v * 0.0000001

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
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux #ori
            #loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * 0.25 
            #loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * 0.0000001
        return loss_dict
