import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import batch_giou_corres, batch_giou_corres_polar, giou_aabb, giou_aabb_polar, get_mask_from_polar_single, get_3D_locs_from_rays, cdf_sequential, get_ray_cls_from_ray
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


class Criterion_Dyco_Ray_Cluster_with_CLS(nn.Module):
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
        super(Criterion_Dyco_Ray_Cluster_with_CLS, self).__init__()

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
            'mask_ray_loss':1,
            'angular_loss_pos':1,
            'angular_loss_neg':1,
            'ray_cls_loss':1,
            'mask2_loss': 1,
        }
    def set_angles_ref(self, angles_ref, width, height):
        self.angles_ref = angles_ref
        self.rays_width = width
        self.rays_height = height
    def set_anchor(self, anchors, norms, cdf_param, range_anchors):
        self.rays_anchors_tensor = anchors
        self.rays_norm_params = norms
        self.rays_anchors_cdf_params = cdf_param
        self.range_anchors = range_anchors
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
       
        losses["pw_giou_loss"] = giou_loss
        losses["pw_conf_loss"] = conf_loss
        
        return losses

    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        mask_ray_logits_list,
        mask_logits_angular_list,
        conf_logits,
        box_preds,
      
        ray_cls_logits,

        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        ray_labels,
       
        batch_size,
        dc_coords_float,
        dc_batch_offset,
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
                mask_logit_angular_b = mask_logits_angular_list[b]
                
            else:
                mask_logit_b = box_preds[b].clone().detach()
            
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]
           
            ray_cls_b = ray_cls_logits[b]


            pred_inds, cls_label, inst_label, box_label, ray_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b], ray_labels[b]
         
          
            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue
            if is_mask:
                mask_logit_pred = mask_logit_b[pred_inds]
                mask_logit_ray_pred = mask_logit_ray_b[pred_inds]
                mask_logit_angular = mask_logit_angular_b[pred_inds]
            
            ray_cls_logit = ray_cls_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
            
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
            if is_mask:

                

                loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                    mask_logit_pred, inst_label, num_gt_batch
                )
                ori_mask = (mask_logit_pred >= 0) *1.0
                FP = (ori_mask == 1) * (inst_label == 0)
                FN = (ori_mask == 0) * (inst_label == 1)
                
                gt_for_mask2 = torch.zeros_like(inst_label)
                gt_for_mask2[FP] = -1
                gt_for_mask2[FN] = 1
                

                loss_dict["mask2_loss"] = loss_dict["mask2_loss"] + compute_dice_loss(
                    mask_logit_angular, inst_label, num_gt_batch
                )
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_angular, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["mask2_loss"] = loss_dict["mask2_loss"] + bce_loss
                aux_loss = (mask_logit_angular[FP].sum()/(FP.sum()+1e-6)) + ((-1*mask_logit_angular[FN]).sum()/(FN.sum()+1e-6))

                loss_dict["mask2_loss"] = loss_dict["mask2_loss"] + aux_loss
                #mask2_loss = F.l1_loss(mask_logit_angular, gt_for_mask2, reduction='none')
                #mask2_loss[gt_for_mask2 == 0] *=0.001
                
                #loss_dict['mask2_loss'] = loss_dict['mask2_loss'] + mask2_loss.sum()/(FP.sum() + FN.sum() + 1e-6)
                
                mask_logit_pred_chk = mask_logit_pred > 0.0
                true_mask = inst_label == 1
                pos_mask_total += mask_logit_pred_chk[true_mask].sum().item()
                num_mask_total += true_mask.sum().item()
               
           
               
            if is_mask:
                
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss
            
                
                #anchors = self.rays_anchors_tensor[cls_label]
                #mask_logit_ray_pred[:,:-3] = mask_logit_ray_pred[:,:-3] + anchors
                
                #ray_label[:,:-3] = ray_label[:,:-3]/10.0
        
                #cdf_target = cdf_sequential(ray_label[:,:-3].cpu().numpy(), self.rays_anchors_cdf_params, cls_label.cpu().numpy())
                
                #ray_label[:,:-3] = torch.tensor(cdf_target).float().cuda()
                
                anchors_ray, ray_cls_label = get_ray_cls_from_ray(ray_label, cls_label, self.range_anchors, return_label=True)
                ray_cls_logit_flatten = ray_cls_logit.reshape(-1,ray_cls_logit.shape[-1])
                
                ray_cls_loss = F.cross_entropy(ray_cls_logit_flatten, ray_cls_label)
                ray_cls = torch.softmax(ray_cls_logit_flatten,1)
                ray_cls = torch.argmax(ray_cls,1)
                acc = (ray_cls == ray_cls_label).sum() / len(ray_cls)
                acc = acc.item()

                loss_dict['ray_cls_loss'] = loss_dict['ray_cls_loss'] + ray_cls_loss 

                ray_loss = F.l1_loss(mask_logit_ray_pred, ray_label - anchors_ray, reduction="none")
                
                #ray_loss = F.l1_loss(mask_logit_ray_pred, ray_label, reduction="none")
                ray_loss = ray_loss.sum() / (num_gt_batch + 1e-6)
                loss_dict["mask_ray_loss"] = loss_dict["mask_ray_loss"] + ray_loss

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
        #if is_aux == False:
        #    print(acc)
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
            mask_logits_angular = model_outputs['mask_logits_angular']
        else:
            mask_logits = []
            mask_ray_logits = []
        conf_logits = model_outputs["conf_logits"]
        box_preds = model_outputs["box_preds"]#Box pred includes anchor positions
     
        ray_cls_pred = model_outputs['ray_cls_pred']

        dc_coords_float = model_outputs['dc_coords_float']
        dc_batch_offset = model_outputs['dc_batch_offsets']
        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]

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
        
        # NOTE main loss
        
        row_indices = gt_dict["row_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]
        ray_labels = gt_dict['ray_labels']
     
        

        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            mask_ray_logits,
            mask_logits_angular,
            conf_logits,
            box_preds,
      
            ray_cls_pred,

            row_indices,
            cls_labels,
            inst_labels,
            box_labels,
            ray_labels,
       
            batch_size,
            dc_coords_float,
            dc_batch_offset,
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
            mask_logits_angular,
            conf_logits,
            box_preds,
         
            ray_cls_pred,

            aux_row_indices,
            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            aux_ray_labels,
           
            batch_size,
            dc_coords_float,
            dc_batch_offset,
            is_mask=is_mask,
            is_aux=True
        )

        coef_aux = 2.0
        
        for k, v in self.loss_weight.items():
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux

        return loss_dict
