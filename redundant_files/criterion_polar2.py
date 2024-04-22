import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import batch_giou_corres, batch_giou_corres_polar, giou_aabb, giou_aabb_polar, batch_giou_corres_polar_exp1, cartesian_to_spherical, spherical_to_cartesian
from torchvision.ops import sigmoid_focal_loss
import numpy as np
import pdb

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
    
    return loss.sum() / (num_boxes + 1e-6)


class Criterion_Polar2(nn.Module):
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
        super(Criterion_Polar2, self).__init__()

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
        self.angles_ref = torch.zeros(1)
        self.register_buffer("empty_weight", empty_weight)
        self.is_anchors = False
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
            "box_loss": 0.9,
            "center_loss": 0.9,
            "giou_loss": 0.5,
        }
        self.matcher_change_dict = {}
    def set_angles_ref(self, angles_ref):
        self.angles_ref = angles_ref
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
        inst_cls_labels,
        inst_sphere,
        offset_labels_sph,
        spherical_mask
    ):
        losses = {}

        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device=semantic_labels.device)
        else:
            weight = None
        
        

        pos_inds = instance_labels != self.ignore_label
        

        spherical_ray_labels = offset_labels_sph[:,:-1]
        spherical_center_labels_offset = offset_labels_sph[:,-1]
       
        rays_labels = cartesian_to_spherical(spherical_ray_labels)
        #chk = spherical_to_cartesian(rays_labels[:,:,3:])
        rays_labels = rays_labels[:,:,3]
        pos_mask = spherical_mask == 1
        

        
        semantic_loss = F.cross_entropy(semantic_scores[pos_mask], semantic_labels[pos_mask], ignore_index=self.ignore_label, weight=weight)
        pred_sem = torch.argmax(semantic_scores[pos_mask],1)
        TP = pred_sem == semantic_labels[pos_mask]

        print('TP: {0}/{1}'.format(TP.sum(), len(TP)))
       

        losses["pw_sem_loss"] = semantic_loss

        total_pos_inds = pos_inds.sum()
        if total_pos_inds == 0:
            offset_loss = 0 * centroid_offset.sum()
            offset_vertices_loss = 0 * corners_offset.sum()
            conf_loss = 0 * box_conf.sum()
            giou_loss = 0 * box_conf.sum()
        else:
            
            offset_loss = (
                F.l1_loss(centroid_offset[pos_mask], centroid_offset_labels[pos_mask], reduction="sum")
                / pos_mask.sum()
            )
           
            offset_vertices_loss = (
                F.l1_loss(corners_offset[pos_mask], rays_labels[pos_mask], reduction="sum") / pos_mask.sum()
            )
            
            if len(angular_offset_labels) > 1 :
               
                angular_offset_sim = F.cosine_similarity(angular_offset[pos_inds], angular_offset_labels[pos_inds])
                angular_offset_loss = 1 - angular_offset_sim
                angular_offset_loss = angular_offset_loss.sum() / pos_mask.sum()#total_pos_inds

                #angular_offset_l1 = F.l1_loss(angular_offset[pos_inds], angular_offset_labels[pos_inds], reduction="sum")
                #angular_offset_l1 = angular_offset_l1 / total_pos_inds

                #angular_offset_loss += angular_offset_l1
                
            else:
                angular_offset_loss = torch.zeros(1).float().cuda()


            '''
           
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
            '''
            ###very small gradient
            conf_labels = torch.zeros(len(box_conf)).float().cuda()
            conf_labels[pos_mask] = 1
            #conf_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction="sum") / total_pos_inds
           
            #conf_loss = sigmoid_focal_loss(box_conf, conf_labels)
            #conf_loss = conf_loss.sum() / pos_mask.sum()
            conf_loss = compute_sigmoid_focal_loss(box_conf, conf_labels, pos_mask.sum())
            max_f = box_conf[pos_mask==False].max()
            min_p = box_conf[pos_mask].min()
            mean_p = box_conf[pos_mask].mean()
            #print('conf max_f: {0} min_p: {1} mean_p : {2}'.format(max_f, min_p, mean_p))
            
            #conf_loss = F.l1_loss(box_conf, conf_labels, reduction="sum") / pos_mask.sum()

        losses["pw_center_loss"] = offset_loss * self.voxel_scale / 50.0
        losses["pw_corners_loss"] = offset_vertices_loss * self.voxel_scale / 50.0
        losses["pw_angular_loss"] = angular_offset_loss
        losses["pw_giou_loss"] = torch.zeros(1).float().cuda()
        losses["pw_conf_loss"] = conf_loss
        
        return losses

    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        conf_logits,
        box_preds,
        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        batch_size,
        col_indices,
        is_mask=True,
        is_aux=False
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

            pred_inds, cls_label, inst_label, box_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b]

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue
            if is_mask:
                mask_logit_pred = mask_logit_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
          
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
            if is_mask:
                loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                    mask_logit_pred, inst_label, num_gt_batch
                )
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
                        
            ####debug due to NaN output--> has to be commented out
            #box_label = torch.clamp(box_label, 1e-6, 1.0)
            ###debug end
            #chk_dist = torch.cdist(box_pred[:,:-3], box_label[:,:-3])
            #chk_dist = torch.argmin(chk_dist,1)
            
            if self.is_anchors:
               
                norm_params = self.rays_norm_params[cls_label]
                box_pred[:,:-3] = ((box_pred[:,:-3])*norm_params[:,:,1]) + norm_params[:,:,0]
                box_pred[:,:-3] += self.rays_anchors_tensor[cls_label]
                #valid_ray_mask = box_label[:,:-3] > 1e-6
                valid_ray_mask = box_label[:,:-3] > -1
                '''
                for idx_label in range(len(box_pred)):
                    
                    #box1_pred = box_pred[idx_label]
                    #box1_label = box_label[idx_label]
                   
                    center_l1 = F.l1_loss(box_pred[idx_label,-3:], box_label[idx_label,-3:])
                    ray_l1 = F.l1_loss(box_pred[idx_label,:-3], box_label[idx_label,:-3])
                    print('box {0} L1 - center : {1} rays : {2}'.format(idx_label, center_l1, ray_l1))
                print('\n')
                '''
             
                loss_dict["box_loss"] = (
                    loss_dict["box_loss"]
                    + (self.voxel_scale / 50.0) * F.l1_loss(box_pred[:,:-3][valid_ray_mask], box_label[:,:-3][valid_ray_mask], reduction="sum") / num_gt_batch
                )
                
            else:
                if is_aux==False:
                    matching = np.concatenate([row_indices[b][:,None], col_indices[b][:,None]],1)
                    '''
                    cnt_change = 0
                    sentence = ''
                    for idx_print in range(len(matching)):
                        gt_num = matching[idx_print,1]
                        refer_name = '{0}_{1}'.format(b, gt_num)
                        pred_loss = F.l1_loss(box_pred[matching[idx_print,1]][:-3], box_label[idx_print,:-3]).detach().item()
                        if refer_name not in self.matcher_change_dict:
                            
                            self.matcher_change_dict[refer_name] = [matching[idx_print,0], box_pred[matching[idx_print,1]],pred_loss]
                        
                        if self.matcher_change_dict[refer_name][0] != matching[idx_print,0]:
                            
                            diff = F.l1_loss(self.matcher_change_dict[refer_name][1][:-3], box_pred[matching[idx_print,1]][:-3]).detach().item()
                        
                            if pred_loss > self.matcher_change_dict[refer_name][2]:
                                cnt_change += 1
                                sentence += ' {0}: {1}->{2} diff: {3} loss: {4} -> {5},'.format(refer_name, self.matcher_change_dict[refer_name][0], matching[idx_print,0], round(diff,2),
                                                                                                round(self.matcher_change_dict[refer_name][2],2),
                                                                                                round(pred_loss,2))
                            self.matcher_change_dict[refer_name] = [matching[idx_print,0], box_pred[matching[idx_print,1]], pred_loss]
                
                    print('negative change in matching: {0}/{1}'.format(cnt_change, len(matching)))
                    '''
                #print('box1 pred: {0} and label: {1} row_idx: {2}'.format(box_pred[10,:-3], box_label[10,:-3], row_indices[0][10]))
                
                if len(box_pred.shape) == 3:

                    arange_num = torch.arange(len(box_pred)).long().cuda()
                    loss_dict["box_loss"] = (
                        loss_dict["box_loss"]
                        + (self.voxel_scale / 50.0) * F.l1_loss(box_pred[arange_num,cls_label,:-3], box_label[:,:-3], reduction="sum") / num_gt_batch
                    )
                else:
                    
                    loss_dict["box_loss"] = (
                        loss_dict["box_loss"]
                        + (self.voxel_scale / 50.0) * F.l1_loss(box_pred[:,:-3], box_label[:,:-3], reduction="sum") / num_gt_batch
                    )
            
            
            if len(box_pred.shape) == 3:

                loss_dict["center_loss"] = (
                    loss_dict["center_loss"]
                    + (self.voxel_scale / 50.0) * F.l1_loss(box_pred[arange_num,cls_label,-3:], box_label[:,-3:], reduction="sum") / num_gt_batch
                )
            else:
                loss_dict["center_loss"] = (
                    loss_dict["center_loss"]
                    + (self.voxel_scale / 50.0) * F.l1_loss(box_pred[:,-3:], box_label[:,-3:], reduction="sum") / num_gt_batch
                )
            
            ##box_label is instance sphere, where the offset is not considered. The value itself is ground-truth
            ##For instance, box_label[-1] is centroid of object, and box_label[:-1] are all 3D points of rays
            
            
            
            
           
            if False and self.angles_ref.shape[0] == 1:
                box_pred = box_pred.reshape(box_pred.shape[0], int(box_pred.shape[1]/3), 3)
                box_label = box_label.reshape(box_label.shape[0], int(box_label.shape[1]/3), 3)
                iou_gt, giou = batch_giou_corres_polar(
                    box_pred,
                        box_label,
                )
            else:
                if len(box_pred.shape) == 3:
                    
                    iou_gt, giou = batch_giou_corres_polar_exp1(
                        box_pred[arange_num,cls_label].detach(),
                            box_label,
                            self.angles_ref
                    )
                else:
                    iou_gt, giou = batch_giou_corres_polar_exp1(
                        box_pred.detach(),
                            box_label,
                            self.angles_ref
                    )
            
            loss_dict["iou_loss"] = (
                loss_dict["iou_loss"] + F.mse_loss(conf_logits_pred, iou_gt, reduction="sum") / num_gt_batch
            )
           
            
            #loss_dict["giou_loss"] = loss_dict["giou_loss"] + torch.sum(1 - giou) / num_gt_batch

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
            
            inst_cls_labels = batch_inputs['instance_cls']
            inst_sphere = batch_inputs['instance_sphere']
            offset_labels_sph = batch_inputs['corners_offset_labels_sph']
            spherical_mask = batch_inputs['spherical_mask']
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
                inst_cls_labels,
                inst_sphere,
                offset_labels_sph,
                spherical_mask
            )
          
            loss_dict.update(point_wise_loss)

            
        return loss_dict

        for k in loss_dict.keys():
            if "pw" in k:
                loss_dict[k] = loss_dict[k] * 0.25
                if torch.isnan(loss_dict[k]).item():
                    pdb.set_trace()
                
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

        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]

        batch_size, n_queries = cls_logits.shape[:2]

        gt_dict, aux_gt_dict, matching_cost = self.matcher.forward_dup(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_inst_mask_arr,
            dup_gt=16,
            angles_ref=self.angles_ref
        )
        #print('matching costs : {0}'.format(np.array(matching_cost).mean()))
        # NOTE main loss

        row_indices = gt_dict["row_indices"]
        col_indices = gt_dict["col_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]
        
        
        
        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            row_indices,
            cls_labels,
            inst_labels,
            box_labels,
            batch_size,
            col_indices,
            is_mask=is_mask
        )

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v
            if torch.isnan(main_loss_dict[k]).item() or torch.isnan(loss_dict[k]).item():
                pdb.set_trace()
        # NOTE aux loss
        
        aux_row_indices = aux_gt_dict["row_indices"]
        aux_inst_labels = aux_gt_dict["inst_labels"]
        aux_cls_labels = aux_gt_dict["cls_labels"]
        aux_box_labels = aux_gt_dict["box_labels"]
        
        aux_main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            aux_row_indices,
            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            batch_size,
            [],
            is_mask=is_mask,
            is_aux=True
        )

        coef_aux = 2.0
        
        for k, v in self.loss_weight.items():
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux
            #if torch.isnan(loss_dict["aux_" + k]).item() or torch.isnan(aux_main_loss_dict[k]).item():
            #    pdb.set_trace()
        #if len(matching_cost) > 0:
        #    loss_dict['matching_cost'] = np.array(matching_cost).mean()
        #else:
        #    loss_dict['matching_cost'] = 1000
        return loss_dict
