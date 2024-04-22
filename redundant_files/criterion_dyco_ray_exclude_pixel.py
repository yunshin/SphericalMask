import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import batch_giou_corres, batch_giou_corres_polar, giou_aabb, giou_aabb_polar, get_mask_from_polar_single, get_3D_locs_from_rays
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


class Criterion_Dyco_Ray(nn.Module):
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
        super(Criterion_Dyco_Ray, self).__init__()

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
            'angular_loss':1,
            'angular_loss_bce':1
        }
    def set_angles_ref(self, angles_ref, width, height):
        self.angles_ref = angles_ref
        self.rays_width = width
        self.rays_height = height
    def set_anchor(self, anchors, norms):
        self.rays_anchors_tensor = anchors
        self.rays_norm_params = norms
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
        mask_ray_logits_list,
        conf_logits,
        box_preds,
        angular_preds,

        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        ray_labels,
        angular_labels,
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

                
            else:
                mask_logit_b = box_preds[b].clone().detach()
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]
            angular_pred_b = angular_preds[b]
            pred_inds, cls_label, inst_label, box_label, ray_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b], ray_labels[b]
            angular_label_b = angular_labels[b]
            dc_coords_float_b = dc_coords_float[dc_batch_offset[b]:dc_batch_offset[b+1]]

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue
            if is_mask:
                mask_logit_pred = mask_logit_b[pred_inds]
                mask_logit_ray_pred = mask_logit_ray_b[pred_inds]
                angular_pred = angular_pred_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
            
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
            if is_mask:
                loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                    mask_logit_pred, inst_label, num_gt_batch
                )
                mask_logit_pred_chk = mask_logit_pred > 0.0
                true_mask = inst_label == 1
                pos_mask_total += mask_logit_pred_chk[true_mask].sum().item()
                num_mask_total += true_mask.sum().item()
               
            if True:
                start_time = time.time()
                angular_label = []
                angular_label_mask = []
                

                box_in_3D = get_3D_locs_from_rays(ray_label[:,:-3], self.angles_ref, ray_label[:,-3:])
                ray_box_label = torch.cat([box_in_3D, ray_label[:,None,-3:]],1)
                ray_box_label = ray_box_label.reshape(len(ray_label),-1)
                
                masks_ray_ = get_mask_from_polar_single(dc_coords_float_b, ray_box_label, self.rays_width, self.rays_height, 
                                                                   is_angular=False, pointwise_angular=torch.ones(len(dc_coords_float_b),2).float().cuda(),
                                                                    sim_ths = 0.8 )
                #mask_logit_ray_pred

                box_in_3D = get_3D_locs_from_rays(mask_logit_ray_pred[:,:-3], self.angles_ref, mask_logit_ray_pred[:,-3:])
                ray_box_pred = torch.cat([box_in_3D, mask_logit_ray_pred[:,None,-3:]],1)
                ray_box_pred = ray_box_label.reshape(len(mask_logit_ray_pred),-1)
                
                masks_ray_pred_ = get_mask_from_polar_single(dc_coords_float_b, ray_box_pred, self.rays_width, self.rays_height, 
                                                                   is_angular=False, pointwise_angular=torch.ones(len(dc_coords_float_b),2).float().cuda(),
                                                                    sim_ths = 0.8 )
                print('time: {0}'.format(time.time() - start_time))
                masks_ray_[masks_ray_==-100] = 0
                masks_ray_pred_[masks_ray_pred_ == -100] = 0
                
                for idx_gt in range(len(inst_label)):
                    
                    mask_angular = (masks_ray_[idx_gt] == 1) * (inst_label[idx_gt]==0)
                    locs_exclude = torch.where(mask_angular)[0]
                    chk_mask = masks_ray_[idx_gt].clone()
                    chk_mask[locs_exclude] = 0
                    chk = torch.allclose(chk_mask==1, inst_label[idx_gt]==1)
                    
                    
                    locs_exclude_pred = torch.where(angular_pred[idx_gt] > 0)[0]
                    chk_mask_pred = masks_ray_pred_[idx_gt].clone()
                    #chk_mask_pred = masks_ray_[idx_gt]
                    
                    chk_mask_pred[locs_exclude_pred] = 0
                    #chk_mask_pred[locs_exclude_pred] = 0
                    
                    overlap = (chk_mask_pred*inst_label[idx_gt]).sum()
                    iou = overlap / (chk_mask_pred.sum() + inst_label[idx_gt].sum() - overlap)
                    #print('{0}: iou {1}'.format(idx_gt, iou))
                    #if chk == False:
                    #    pdb.set_trace()
                    
                    angular_label_mask.append(mask_angular[None,:]*1.0)
                
                angular_label_mask = torch.cat(angular_label_mask)

                loss_dict["angular_loss"] = loss_dict["angular_loss"] + compute_dice_loss(
                    angular_pred, angular_label_mask, num_gt_batch
                )
                #loss_dict["angular_loss"] = loss_dict["angular_loss"] + compute_dice_loss(
                #    angular_pred, inst_label, num_gt_batch
                #)
               
                mask_logit_pred_chk = angular_pred > 0.0
                true_mask = angular_label_mask == 1
                #true_mask = inst_label == 1
                pos_angular_samples_total += mask_logit_pred_chk[true_mask].sum().item()
                num_angular_samples_total += true_mask.sum().item()
            

                angular_bce_loss = F.binary_cross_entropy_with_logits(angular_pred, angular_label_mask, reduction="none")
               
                angular_bce_loss = angular_bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["angular_loss_bce"] = loss_dict["angular_loss_bce"] + angular_bce_loss
                
                '''
                angular_labels_ = angular_label_b[None,:].repeat(angular_pred.shape[0],1,1)
                
                
                angular_mask = angular_label_mask.reshape(-1)
                angular_mask = angular_mask == 1
                num_inst_angular = 10
                cos_sim = F.cosine_similarity(angular_pred.reshape(-1,2)[angular_mask], angular_labels_.reshape(-1,2)[angular_mask]*0.1)
                sim_loss = (1 - cos_sim)
                sim_loss = sim_loss.sum() / angular_mask.sum()

                angular_loss_l1 = F.l1_loss(angular_pred.reshape(-1,2)[angular_mask], angular_labels_.reshape(-1,2)[angular_mask]*0.1, reduction="none")
                angular_loss_l1 = angular_loss_l1.sum() / (angular_mask.sum() + 1e-6)
                
                pos_angular_samples = cos_sim > 0.8
                loss_dict['angular_loss'] = loss_dict['angular_loss'] + sim_loss
                loss_dict['angular_loss_l1'] = loss_dict['angular_loss_l1'] + angular_loss_l1
                '''

                
                
            if is_mask:
                
                bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
                bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
                loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss
            
                
                #anchors = self.rays_anchors_tensor[cls_label]
                #mask_logit_ray_pred[:,:-3] = mask_logit_ray_pred[:,:-3] + anchors
                ray_loss = F.l1_loss(mask_logit_ray_pred, ray_label, reduction="none")
                
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
        angular_pred = model_outputs['angular_pred']

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
        angular_labels = gt_dict['angular_labels']
        
        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            mask_ray_logits,
            conf_logits,
            box_preds,
            angular_pred,
            row_indices,

            cls_labels,
            inst_labels,
            box_labels,
            ray_labels,
            angular_labels,
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
            conf_logits,
            box_preds,
            angular_pred,

            aux_row_indices,
            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            aux_ray_labels,
            aux_angular_labels,
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
