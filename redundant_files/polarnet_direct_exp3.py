import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import functools
import math
from ..ops import voxelization
from ..util import cuda_cast, rle_encode_gpu, rle_encode_gpu_batch
from .aggregator import LocalAggregator
from .blocks import MLP, GenericMLP, ResidualBlock, UBlock, conv_with_kaiming_uniform
from .model_utils import (
    custom_scatter_mean,
    get_batch_offsets,
    get_instance_info,
    get_instance_info_polarnet_exp3,
    get_spp_gt,
    get_subsample_gt,
    nms,
    random_downsample,
    superpoint_align,
    get_cropped_instance_label,
    batch_giou_cross_polar,
    get_mask_from_polar,
    get_mask_from_polar_single,
    get_mask_iou,
    get_3D_locs_from_rays,
    analyze_pred_ray,
    manual_iou,
    get_iou,
    batch_giou_cross_polar_exp1,
    matrix_nms_with_ious,
    standard_nms_with_iou,
    cartesian_to_spherical,
    get_instance_info_dyco_ray
)
import numpy as np
import os
import pickle
import pdb
import time
class PolarNet_Direct_EXP3(nn.Module):
    def __init__(
        self,
        is_mask=True,
        model_name='polarnet_direct_exp2',
        channels=32,
        num_blocks=7,
        semantic_only=False,
        semantic_classes=20,
        instance_classes=18,
        semantic_weight=None,
        sem2ins_classes=[],
        ignore_label=-100,
        with_coords=True,
        instance_head_cfg=None,
        test_cfg=None,
        fixed_modules=[],
        criterion=None,
        dataset_name="scannetv2",
        trainall=False,
        voxel_scale=50,
        use_spp_pool=True,
        filter_bg_thresh=0.1,
        iterative_sampling=True,
        rays_width=3,
        rays_height=2
    ):
        super().__init__()
        self.is_analyze = False
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.instance_head_cfg = instance_head_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.voxel_scale = voxel_scale
        self.rays_width = rays_width
        self.rays_height = rays_height

        self.criterion = criterion
        self.dataset_name = dataset_name

        self.label_shift = semantic_classes - instance_classes

        self.trainall = trainall

        self.use_spp_pool = use_spp_pool
        self.filter_bg_thresh = filter_bg_thresh

        # NOTE iterative sampling
        self.iterative_sampling = iterative_sampling

        in_channels = 6 if with_coords else 3

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # NOTE backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # # NOTE point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
      
        #self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=3)
        #self.angular_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=3)
       
        self.is_mask = is_mask
        self.is_debug = False
        # NOTE BBox
        
        #self.offset_vertices_linear = MLP(channels, self.rays_width*self.rays_height, norm_fn=norm_fn, num_layers=2)
        #self.box_conf_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=3)
        '''
        self.semantic_linear = GenericMLP(
                input_dim=channels,
                hidden_dims=[channels, channels],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=semantic_classes,
            )
        '''
        self.offset_linear = GenericMLP(
                input_dim=channels,
                hidden_dims=[channels, channels],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=3,
            )
        self.angular_linear = GenericMLP(
                input_dim=channels,
                hidden_dims=[channels, channels],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=2,
            )
        self.box_conf_linear = GenericMLP(
                input_dim=channels,
                hidden_dims=[channels, channels],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=1,
            )
        self.offset_vertices_linear = GenericMLP(
                input_dim=channels,
                hidden_dims=[channels, channels],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=self.rays_width*self.rays_height,
            )
        xy_angles = torch.arange(0,360, int(360/self.rays_width))
        yz_angles = torch.arange(0,180, int(180/self.rays_height))
        self.angles_ref = torch.zeros((self.rays_height* self.rays_width,2))
        angle_cnt = 0
        for idx in range(self.rays_height):

            for idx2 in range(self.rays_width):
                self.angles_ref[angle_cnt,0] = yz_angles[idx]
                self.angles_ref[angle_cnt,1] = xy_angles[idx2]
                
                angle_cnt += 1
        #pdb.set_trace()
        self.criterion.set_angles_ref(self.angles_ref)
        
        '''
        if not self.semantic_only:
            self.point_aggregator1 = LocalAggregator(
                mlp_dim=self.channels,
                n_sample=instance_head_cfg.n_sample_pa1,
                radius=0.2 * instance_head_cfg.radius_scale,
                n_neighbor=instance_head_cfg.neighbor,
                n_neighbor_post=instance_head_cfg.neighbor * 2#,
                #num_rays_width=self.rays_width,
                #num_rays_height=self.rays_height
            )

            self.point_aggregator2 = LocalAggregator(
                mlp_dim=self.channels * 2,
                n_sample=instance_head_cfg.n_queries,
                radius=0.4 * instance_head_cfg.radius_scale,
                n_neighbor=instance_head_cfg.neighbor,
                n_neighbor_post=instance_head_cfg.neighbor#,
                #num_rays_width=self.rays_width,
                #num_rays_height=self.rays_height
            )

            self.inst_shared_mlp = GenericMLP(
                input_dim=self.channels * 4,
                hidden_dims=[self.channels * 4],
                output_dim=instance_head_cfg.dec_dim,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_use_activation=True,
                output_use_norm=True,
            )

            self.inst_sem_head = GenericMLP(
                input_dim=instance_head_cfg.dec_dim,
                hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=self.instance_classes + 1,
            )

            self.inst_conf_head = GenericMLP(
                input_dim=instance_head_cfg.dec_dim,
                hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=1,
            )
            
            if rays_width == 0:
                self.inst_box_head = GenericMLP(
                    input_dim=instance_head_cfg.dec_dim,
                    hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                    norm_fn_name="bn1d",
                    activation="relu",
                    use_conv=True,
                    output_dim=6,  # xyz_min, xyz_max
                )
            else:
                self.inst_box_head = GenericMLP(
                    input_dim=instance_head_cfg.dec_dim,
                    hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                    norm_fn_name="id",
                    activation="relu",
                    use_conv=True,
                    output_dim=rays_width*rays_height*self.instance_classes ,  # xyz_min, xyz_max
                )
                
                self.inst_box_center_head = GenericMLP(
                    input_dim=instance_head_cfg.dec_dim,
                    hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                    norm_fn_name="id",
                    activation="relu",
                    use_conv=True,
                    output_dim=3#*self.instance_classes,  # xyz_min, xyz_max
                )
                

            # NOTE dyco
            if self.is_mask:
                self.init_dyco()
        
        self.init_weights()
        '''
        self.best_iou = torch.zeros(27).float().cuda()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        if "input_conv" in self.fixed_modules and "unet" in self.fixed_modules:
            self.freeze_backbone = True
        else:
            self.freeze_backbone = False
        #self.freeze_backbone = True
        #with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_wo_zero.pkl', 'rb') as f:
        
        if os.path.exists('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_{0}_{1}.pkl'.format(rays_height, rays_width)):
            with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_{0}_{1}.pkl'.format(rays_height, rays_width), 'rb') as f:
                self.rays_anchors = pickle.load(f)
            #with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_wo_zero_norm_params.pkl', 'rb') as f:
            with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_wo_zero_norm_params_{0}_{1}.pkl'.format(rays_height, rays_width), 'rb') as f:
                self.norm_params = pickle.load(f)
            
            
            self.rays_anchors_tensor = torch.zeros(self.instance_classes + 1, self.rays_width*self.rays_height).float().cuda()
            self.rays_norm_params = torch.zeros(self.instance_classes + 1, self.rays_width*self.rays_height,2).float().cuda()
            self.rays_norm_params[:,:,0] = 0
            self.rays_norm_params[:,:,1] = 1
        
            for key in self.rays_anchors:
                
                self.rays_anchors_tensor[key] = torch.tensor(self.rays_anchors[key][:-3]).float().cuda()
                
                self.rays_norm_params[key,:,0] = torch.tensor(self.norm_params[key][0][:-3]).float().cuda()
                self.rays_norm_params[key,:,1] = torch.tensor(self.norm_params[key][1][:-3]).float().cuda()
            
            self.criterion.set_anchor(self.rays_anchors_tensor, self.rays_norm_params)
        
    def init_dyco(self):
        conv_block = conv_with_kaiming_uniform("BN", activation=True)

        self.mask_dim_out = 32

        mask_tower = [
            conv_block(self.channels, self.channels),
            conv_block(self.channels, self.channels),
            conv_block(self.channels, self.channels),
            nn.Conv1d(self.channels, self.mask_dim_out, 1),
        ]
        self.add_module("mask_tower", nn.Sequential(*mask_tower))

        #weight_nums = [(self.mask_dim_out + 3 + 3) * self.mask_dim_out, self.mask_dim_out * 1]
        weight_nums = [(self.mask_dim_out + self.rays_width*(self.rays_height)*3 + 3 + 3) * self.mask_dim_out, self.mask_dim_out * 1]
        bias_nums = [self.mask_dim_out, 1]

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        # NOTE convolution before the condinst take place (convolution num before the generated parameters take place)
        inst_mask_head = [
            conv_block(self.instance_head_cfg.dec_dim, self.instance_head_cfg.dec_dim),
            conv_block(self.instance_head_cfg.dec_dim, self.instance_head_cfg.dec_dim),
        ]
        controller_head = nn.Conv1d(self.instance_head_cfg.dec_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(controller_head.weight, std=0.01)
        torch.nn.init.constant_(controller_head.bias, 0)
        inst_mask_head.append(controller_head)
        self.add_module("inst_mask_head", nn.Sequential(*inst_mask_head))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
    def refine_cls(self, indices, voxel_semantic_labels):
        indice_mask = []
        for idx_indice in range(len(indices)):
            if voxel_semantic_labels[indices[idx_indice]] != -100:
                indice_mask.append(1)
            else:
                indice_mask.append(0)
        
        indice_mask = np.array(indice_mask)
        indice_mask = indice_mask == 1
        indices = indices[indice_mask]
        
        category_labels = voxel_semantic_labels[indices]
        return indices, category_labels, indice_mask
    
    def refine_conf(self, voxel_box_conf_, spherical_mask):

        voxel_box_conf_[:] = 0
        voxel_box_conf_[spherical_mask == 1] = 1
        return voxel_box_conf_
    
    def refine_samples(self, box_preds, box_gt, spherical_mask):
        ori_locs = torch.where(spherical_mask==1)[0]
        box_preds_sph = box_preds[spherical_mask==1]

        for idx in range(len(box_gt)):


            box_dist = torch.cdist( box_gt[idx,-1], box_preds_sph[:,-1])
            box_dist, box_dist_indice = torch.min(box_dist, 1)
        pdb.set_trace()
    def refine_center(self, voxel_box_preds_nms, category_labels, corners_offset_labels_sph, instance_cls, voxel_coord_float, center_refine=True, ray_refine=False):

        for idx in range(len(voxel_box_preds_nms)):
            cls_label = category_labels[idx]
            box = voxel_box_preds_nms[idx]
            center = box[-3:]
            box = box[:-3]
            center_label = corners_offset_labels_sph[idx,-1]
            center_refined = voxel_coord_float[idx] + center_label

            
            if center_refine:
                center = center_refined
            if ray_refine:
                
                #box = torch.tensor(self.rays_anchors[cls_label.item()][:-3]).float().cuda()
                box = corners_offset_labels_sph[idx,:-1,0]
            voxel_box_preds_nms[idx,:-3] = box
            voxel_box_preds_nms[idx,-3:] = center
            
        return voxel_box_preds_nms

    def forward(self, batch, return_loss=False, epoch=0):
        if return_loss:
            return self.forward_train(**batch, epoch=epoch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(
        self,
        batch_idxs,
        voxel_coords,
        p2v_map,
        v2p_map,
        coords_float,
        feats,
        semantic_labels,
        instance_labels,
        spps,
        spatial_shape,
        batch_size,
        epoch,
        **kwargs
    ):
        # NOTE voxelization
        voxel_semantic_labels = semantic_labels[p2v_map[:, 1].long()]
        voxel_instance_labels = instance_labels[p2v_map[:, 1].long()]
        voxel_instance_labels = get_cropped_instance_label(voxel_instance_labels)

        voxel_spps = spps[p2v_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        voxel_coords_float = voxelization(coords_float, p2v_map)
        
        instance_cls, instance_box, instance_sphere, centroid_offset_labels, pointwise_angular_labels, corners_offset_labels, corners_offset_labels_sph, rays_labels, spherical_mask, voxel_semantic_labels  = get_instance_info_polarnet_exp3(
            voxel_coords_float, voxel_instance_labels, voxel_semantic_labels, self.rays_width, self.rays_height, self.angles_ref, label_shift=self.label_shift
        )
      
        ###################
      
        batch_inputs = dict(
            semantic_labels=voxel_semantic_labels,
            instance_labels=voxel_instance_labels,
            instance_cls=instance_cls,
            instance_sphere=instance_sphere,
            corners_offset_labels_sph=corners_offset_labels_sph,
            spherical_mask=spherical_mask
        )

        model_outputs = {}

        voxel_output_feats, voxel_batch_idxs = self.forward_backbone(
            feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map
        )
        voxel_batch_offsets = get_batch_offsets(voxel_batch_idxs, batch_size)

        (
            voxel_semantic_scores,
            voxel_centroid_offset,
            voxel_corners_offset,
            voxel_angular_offset,
            voxel_box_conf,
        ) = self.forward_pointwise_head(voxel_output_feats)
        #pdb.set_trace()
        #####This was commented out
        #voxel_centroid_offset += voxel_coords_float
        ##################
        voxel_box_preds = voxel_corners_offset
        #voxel_box_preds = voxel_corners_offset + voxel_coords_float[:,None,:].repeat(1,(self.rays_width*self.rays_height)+1,1)  # n_points, 6
        #voxel_box_preds = voxel_corners_offset + voxel_coords_float.repeat(1, 2)  # n_points, 6
        

        #voxel_box_preds = voxel_box_preds.view(voxel_box_preds.shape[0], (self.rays_width*self.rays_height)*3 + 3)
        
        #instance_box = instance_sphere
        
        instance_box = torch.cat([rays_labels,instance_sphere[:,-1][:,:,None]],1)[:,:,0]
        instance_sphere = instance_sphere.view(instance_sphere.shape[0], (self.rays_width*self.rays_height)*3 + 3)
        if self.semantic_only or self.trainall:
            
            batch_inputs.update(
                dict(
                    coords_float=voxel_coords_float,
                    centroid_offset_labels=centroid_offset_labels,
                    corners_offset_labels=corners_offset_labels,
                    angular_offset_labels=pointwise_angular_labels
                )
            )

            model_outputs.update(
                dict(
                    semantic_scores=voxel_semantic_scores,
                    centroid_offset=voxel_centroid_offset,
                    corners_offset=voxel_corners_offset,
                    angular_offset=voxel_angular_offset,
                    box_conf=voxel_box_conf,
                )
            )

            if self.semantic_only:
                # NOTE cal loss
                losses = self.criterion(batch_inputs, model_outputs)

                return self.parse_losses(losses)

        # NOTE Point Aggregator
        voxel_semantic_scores_sm = F.softmax(voxel_semantic_scores, dim=1)
        spp_semantic_scores_sm = custom_scatter_mean(
            voxel_semantic_scores_sm, voxel_spps, dim=0, pool=self.use_spp_pool
        )
        spp_object_conditions = torch.any(
            spp_semantic_scores_sm[:, self.label_shift :] >= self.filter_bg_thresh, dim=-1
        )
        object_conditions = spp_object_conditions[voxel_spps]
        object_idxs = torch.nonzero(object_conditions).view(-1)
       
        if len(object_idxs) <= 100:
            
            loss_dict = {
                "placeholder": torch.tensor(0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float)
            }
            return self.parse_losses(loss_dict)
       
        voxel_batch_idxs_ = voxel_batch_idxs[object_idxs]
        voxel_output_feats_ = voxel_output_feats[object_idxs]
        voxel_coords_float_ = voxel_coords_float[object_idxs]
        voxel_box_preds_ = voxel_box_preds[object_idxs]
        voxel_batch_offsets_ = get_batch_offsets(voxel_batch_idxs_, batch_size)
        
        
        #torch.Size([531162, 6])
        ###Sampling starts
        '''
        query_locs, query_feats, query_boxes, query_inds1 = self.point_aggregator1(
            voxel_coords_float_,
            voxel_output_feats_,
            voxel_box_preds_,
            batch_offsets=voxel_batch_offsets_,
            batch_size=batch_size,
            sampled_before=False,
        )
       
        query_locs, query_feats, query_boxes, query_inds = self.point_aggregator2(
            query_locs, query_feats, query_boxes, batch_offsets=None, batch_size=batch_size, sampled_before=True
        )
        
        ###Sampling done
        # -------------------------------
        
        # NOTE Dynamic conv
        if self.use_spp_pool:
            dc_coords_float, dc_output_feats, dc_box_preds, dc_batch_offsets = self.spp_pool(
                voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets
            )

            # NOTE Get GT and loss for batch
            dc_inst_mask_arr = get_spp_gt(
                voxel_instance_labels,
                voxel_spps,
                instance_cls,
                instance_box,
                voxel_batch_offsets,
                batch_size,
                pool=self.use_spp_pool,
            )
            
        else:
            idxs_subsample = random_downsample(voxel_batch_offsets_, batch_size, n_subsample=15000)
            dc_coords_float = voxel_coords_float_[idxs_subsample]
            dc_box_preds = voxel_box_preds_[idxs_subsample]
            dc_output_feats = voxel_output_feats_[idxs_subsample]
            dc_batch_offsets = get_batch_offsets(voxel_batch_idxs_[idxs_subsample], batch_size)

            subsample_idxs = object_idxs[idxs_subsample]
            
            dc_inst_mask_arr = get_subsample_gt(
                voxel_instance_labels, subsample_idxs, instance_cls, instance_box, dc_batch_offsets, batch_size
            )

       
        if self.is_mask:
            dc_mask_features = self.mask_tower(torch.unsqueeze(dc_output_feats, dim=2).permute(2, 1, 0)).permute(2, 1, 0)
            cls_logits, mask_logits, conf_logits,  box_preds = self.forward_head(
                query_feats, query_locs, dc_mask_features, dc_coords_float, dc_box_preds, dc_batch_offsets
            )
        else:
            cls_logits, conf_logits,box_preds = self.forward_head(
                query_feats, query_locs, [], dc_coords_float, dc_box_preds, dc_batch_offsets
            )
       
        # -------------------------------
        
        
        
        if self.is_mask:
            model_outputs.update(
                dict(
                    dc_inst_mask_arr=dc_inst_mask_arr,
                    dc_batch_offsets=dc_batch_offsets,
                    cls_logits=cls_logits,
                    mask_logits=mask_logits,
                    conf_logits=conf_logits,
                    box_preds=box_preds,
                    query_locs=query_locs
                )
            )
        else:
            model_outputs.update(
                dict(
                    dc_inst_mask_arr=dc_inst_mask_arr,
                    dc_batch_offsets=dc_batch_offsets,
                    cls_logits=cls_logits,
                    conf_logits=conf_logits,
                    box_preds=box_preds,
                    query_locs=query_locs
                )
            )
        '''
        losses = self.criterion(batch_inputs, model_outputs,is_mask=self.is_mask)
        
        # -------------------------------

        return self.parse_losses(losses)

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        
        return loss, losses

    @cuda_cast
    def forward_test(
        self,
        batch_idxs,
        voxel_coords,
        p2v_map,
        v2p_map,
        coords_float,
        feats,
        semantic_labels,
        instance_labels,
        spps,
        spatial_shape,
        batch_size,
        scan_ids,
        **kwargs
    ):
        #pdb.set_trace()
        #assert batch_size == 1

        voxel_spps = spps[p2v_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        ret = dict(
            scan_id=scan_ids[0],
            semantic_labels=semantic_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy(),
            coords_float=coords_float.cpu().numpy(),
        )

        voxel_coords_float = voxelization(coords_float, p2v_map)
        
        if self.is_mask == False:
            voxel_semantic_labels = semantic_labels[p2v_map[:, 1].long()]
            voxel_instance_labels = instance_labels[p2v_map[:, 1].long()]
            voxel_instance_labels = get_cropped_instance_label(voxel_instance_labels)
            
            
            if self.is_analyze:
                instance_cls, instance_box, instance_sphere, centroid_offset_labels, pointwise_angular_labels, corners_offset_labels, corners_offset_labels_sph, rays_labels, spherical_mask, voxel_semantic_labels  = get_instance_info_dyco_ray(
                    voxel_coords_float, voxel_instance_labels, voxel_semantic_labels, self.rays_width, self.rays_height, self.angles_ref, label_shift=self.label_shift
                )
                '''
                instance_cls, instance_box, instance_sphere, centroid_offset_labels, pointwise_angular_labels, corners_offset_labels, rays_gt_list, voxel_semantic_labels = get_instance_info_polarnet_exp3(
                    voxel_coords_float, voxel_instance_labels, voxel_semantic_labels, self.rays_width, self.rays_height, self.angles_ref, label_shift=self.label_shift, is_analyze=self.is_analyze
                )
                '''
                
                return instance_cls, instance_box, instance_sphere, centroid_offset_labels, pointwise_angular_labels, corners_offset_labels, rays_labels

            instance_cls, instance_box, instance_sphere, centroid_offset_labels, pointwise_angular_labels, corners_offset_labels, corners_offset_labels_sph, rays_gt, spherical_mask, voxel_semantic_labels = get_instance_info_polarnet_exp3(
                voxel_coords_float, voxel_instance_labels, voxel_semantic_labels, self.rays_width, self.rays_height, self.angles_ref, label_shift=self.label_shift
            )
            self.instance_cls = instance_cls
            self.instance_sphere = instance_sphere
            self.p2v_map = p2v_map
        voxel_output_feats, voxel_batch_idxs = self.forward_backbone(
            feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map, x4_split=self.test_cfg.x4_split
        )
        
        voxel_batch_offsets = get_batch_offsets(voxel_batch_idxs, batch_size)

        (
            voxel_semantic_scores,
            voxel_centroid_offset,
            voxel_corners_offset,
            voxel_angular_offset,
            voxel_box_conf,
        ) = self.forward_pointwise_head(voxel_output_feats)

        voxel_centroid_offset += voxel_coords_float

        #voxel_box_preds = voxel_corners_offset + voxel_coords_float.repeat(1, 2)  # n_points, 6
        #voxel_box_preds = voxel_corners_offset + voxel_coords_float[:,None,:].repeat(1,(self.rays_width*self.rays_height)+1,1) 
        voxel_box_preds = voxel_corners_offset
        #voxel_box_preds = voxel_box_preds.view(voxel_box_preds.shape[0], self.rays_width*(self.rays_height)*3 + 3)
        voxel_semantic_scores_pred = torch.argmax(voxel_semantic_scores, dim=1)  # N_points
        voxel_semantic_scores_sm = F.softmax(voxel_semantic_scores, dim=1)
        
       
        if self.is_debug:
            voxel_semantic_scores_sm[:,self.label_shift] = 0.3
        

        spp_semantic_scores_sm = custom_scatter_mean(
            voxel_semantic_scores_sm,
            voxel_spps[:, None].expand(-1, voxel_semantic_scores_sm.shape[-1]),
            pool=self.use_spp_pool,
        )

        spp_object_conditions = torch.any(
            spp_semantic_scores_sm[:, self.label_shift :] >= self.filter_bg_thresh, dim=-1
        )
        object_conditions = spp_object_conditions[voxel_spps]
      
        if object_conditions.sum() == 0:
            ret.update(dict(pred_instances=[]))
            return ret

        # NOTE Local Aggregator

        object_idxs = torch.nonzero(object_conditions).view(-1)
        voxel_batch_idxs_ = voxel_batch_idxs[object_idxs]
        voxel_output_feats_ = voxel_output_feats[object_idxs]
        voxel_coords_float_ = voxel_coords_float[object_idxs]
        voxel_centroid_offset_ = voxel_centroid_offset[object_idxs]
        voxel_box_preds_ = voxel_box_preds[object_idxs]
        voxel_box_conf_ = torch.sigmoid(voxel_box_conf[object_idxs])
        voxel_semantic_scores_pred = voxel_semantic_scores_pred[object_idxs]
        voxel_semantic_scores_sm = voxel_semantic_scores_sm[object_idxs]
        spherical_mask = spherical_mask[object_idxs]
        voxel_semantic_labels = voxel_semantic_labels[object_idxs]
        corners_offset_labels_sph = corners_offset_labels_sph[object_idxs]
        centroid_offset_labels = centroid_offset_labels[object_idxs]
        

        voxel_spps_ = voxel_spps[object_idxs]
        voxel_spps_ = torch.unique(voxel_spps_, return_inverse=True)[1]

        dc_coords_float, dc_output_feats, dc_box_preds, dc_batch_offsets = self.spp_pool(
            voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets
        )
        #voxel_coords_float_[spherical_mask]
        voxel_batch_offsets_ = get_batch_offsets(voxel_batch_idxs_, batch_size)
      
        voxel_box_preds_ = torch.cat([voxel_box_preds_, voxel_centroid_offset_],1)
        
        ###
        #voxel_box_conf_[:] = 0
        #voxel_box_conf_[spherical_mask==1] = 1
        ###
        
        if self.is_debug == False:

            voxel_box_conf_ = self.refine_conf(voxel_box_conf_, spherical_mask)

            conf_sort_indice = torch.argsort(voxel_box_conf_, descending=True)
            conf_sort_indice = conf_sort_indice[:1024]
           
            pos_mask = voxel_box_conf_[conf_sort_indice] > 0.0
            conf_sort_indice = conf_sort_indice[pos_mask]
            
            iou_, _ = batch_giou_cross_polar_exp1(voxel_box_preds_[conf_sort_indice], voxel_box_preds_[conf_sort_indice], self.angles_ref)
            categories = voxel_semantic_scores_pred[conf_sort_indice]
            voxel_box_conf_ = voxel_box_conf_[conf_sort_indice]
            voxel_box_preds_ = voxel_box_preds_[conf_sort_indice]
            voxel_semantic_labels = voxel_semantic_labels[conf_sort_indice]
            voxel_semantic_scores_sm = voxel_semantic_scores_sm[conf_sort_indice]
            spherical_mask = spherical_mask[conf_sort_indice]
            voxel_coords_float_sampled = voxel_coords_float_[conf_sort_indice]
            corners_offset_labels_sph = corners_offset_labels_sph[conf_sort_indice]
            #voxel_coords_float_ = voxel_coords_float_[conf_sort_indice]

            categories_nms, voxel_box_conf_nms, voxel_box_preds_nms, indices = standard_nms_with_iou(categories, voxel_box_conf_, voxel_box_preds_,iou_, threshold=0.9)
            voxel_box_conf_ = voxel_box_conf_[indices]
            
            
            dist_ = torch.cdist( instance_sphere[:,-1], voxel_box_preds_nms[:,-3:])
            dist_, _  = torch.min(dist_,1)
            print('dist_: {0}'.format(dist_))
            ####Label gt
            
            ###
            refine_cls = False
            if refine_cls :
                indices, category_labels, indice_mask = self.refine_cls(indices, voxel_semantic_labels)
                voxel_box_preds_nms = voxel_box_preds_nms[indice_mask]
                voxel_box_conf_nms = voxel_box_conf_nms[indice_mask]
                categories_nms = categories_nms[indice_mask]
            ###
            
            voxel_coords_float_sampled = voxel_coords_float_sampled[indices]
            voxel_semantic_scores_sm = voxel_semantic_scores_sm[indices]
            spherical_mask = spherical_mask[indices]
           
            ###using pred
            if refine_cls == False:
                category_labels = torch.argmax(voxel_semantic_scores_sm,1)
          
            ###
            voxel_box_preds_nms[:,:-3] = voxel_box_preds_nms[:,:-3] + self.rays_anchors_tensor[category_labels]


            corners_offset_labels_sph = corners_offset_labels_sph[indices]
            
            voxel_box_preds_nms = self.refine_center(voxel_box_preds_nms, category_labels, corners_offset_labels_sph, instance_cls, voxel_coords_float_sampled)
            
            box_in_3D = get_3D_locs_from_rays(voxel_box_preds_nms[:,:-3], self.angles_ref, voxel_box_preds_nms[:,-3:])
            box_preds = torch.cat([box_in_3D, voxel_box_preds_nms[:,None,-3:]],1)
          
            box_preds = box_preds.reshape(len(box_preds),-1)
            
            

        if self.is_debug:
            
            rays_label = corners_offset_labels_sph[spherical_mask==1]
            rays_label = rays_label[:,:-1,0]
            category_label = voxel_semantic_labels[spherical_mask==1]
            
            '''
            rays_anchors = torch.zeros(len(category_label), self.rays_height*self.rays_width).float().cuda()
            for idx_anchor in range(len(category_label)):
                
                anchor = self.rays_anchors[category_label[idx_anchor].item()][:-3]
                rays_anchors[idx_anchor] = torch.tensor(anchor).float().cuda()
            rays_label = rays_anchors
            '''
            ##############
            
            centroid_offset_labels = corners_offset_labels_sph[:,-1]
            
            voxel_centroid_offset_label = centroid_offset_labels + voxel_coords_float_
            #rays_label = cartesian_to_spherical(rays_label)
            #rays_label = rays_label[:,:,3]
          
            box_in_3D = get_3D_locs_from_rays(rays_label, self.angles_ref, voxel_centroid_offset_label[spherical_mask==1])
            box_preds_label = torch.cat([box_in_3D, voxel_centroid_offset_label[spherical_mask==1][:,None,:]],1)
            
            box_preds_label = box_preds_label.reshape(len(box_preds_label),-1)
            #rays_label = torch.cat([rays_label, voxel_coords_float_[spherical_mask==1][:,None,:]],1)
            
            masks_ray_, masks_angles_ = get_mask_from_polar_single(voxel_coords_float, box_preds_label, self.rays_width, self.rays_height, 
                                                                    is_angular=True, pointwise_angular=pointwise_angular_labels,
                                                                        sim_ths = 0.90 )
            
            closest_label = torch.cat([torch.arange(len(category_label)).long().cuda()[:,None], category_label[:,None]],1 )
            voxel_semantic_scores_sm_label = torch.zeros(len(box_preds_label),voxel_semantic_scores_sm.shape[1]).float().cuda()
            voxel_semantic_scores_sm_label[:, :] = -3
            voxel_semantic_scores_sm_label[closest_label[:,0], closest_label[:,1]] = 5
            voxel_box_conf_label = torch.ones(len(closest_label)).float().cuda()
            
            voxel_semantic_scores_sm = voxel_semantic_scores_sm_label
            voxel_box_conf_nms = voxel_box_conf_label
            voxel_box_preds_nms = box_preds_label
        else:
            
            masks_ray_, masks_angles_ = get_mask_from_polar_single(dc_coords_float, box_preds, self.rays_width, self.rays_height, 
                                                                    is_angular=True, pointwise_angular=voxel_angular_offset,
                                                                        sim_ths = 0.90 )
            
            masks_ray_ = masks_ray_[:, voxel_spps]
            #pdb.set_trace()
            max_gt2pred, iou_gt2pred, closest_label, cls_accs = manual_iou(masks_ray_, instance_labels, v2p_map, p2v_map[:,1], categories_nms, instance_cls, p2v=True)
            #max_gt2pred, iou_gt2pred, closest_label, cls_accs = manual_iou(masks_ray_, instance_labels, v2p_map, p2v_map[:,1], category_labels, instance_cls, p2v=True)
            #print(voxel_semantic_labels[spherical_mask==1])
            #print(closest_label)
            closest_label = torch.cat([torch.arange(len(closest_label)).long().cuda()[:,None], category_labels[:,None]],1 )
            voxel_semantic_scores_sm[:,:] = -3
            voxel_semantic_scores_sm[closest_label[:,0], closest_label[:,1]] = 5
           
        mask_pure_ray = []
        mask_angles = []
        mask_pure_ray.append(masks_ray_)
        mask_angles.append(masks_angles_)

        

        cls_logits_final = []
        conf_logits_final = []
        box_preds_final = []
        cls_logits_final.append(voxel_semantic_scores_sm)
        conf_logits_final.append(voxel_box_conf_nms)
        box_preds_final.append(voxel_box_preds_nms)
        
        
        
        
        '''
        if self.is_mask:
            dc_mask_features = self.mask_tower(torch.unsqueeze(dc_output_feats, dim=2).permute(2, 1, 0)).permute(2, 1, 0)
        # -------------------------------

        if self.dataset_name != "s3dis":
            query_locs1, query_feats1, query_boxes1, query_inds1 = self.point_aggregator1(
                voxel_coords_float_,
                voxel_output_feats_,
                voxel_box_preds_,
                batch_offsets=voxel_batch_offsets_,
                batch_size=batch_size,
                sampled_before=False,
            )
        else:  # NOTE subsample for s3dis as in testing, density is higher than training
            n_samples = ((voxel_coords[:, 0] == 0) | (voxel_coords[:, 0] == 1)).sum()
            s_cond = object_idxs <= n_samples
            voxel_batch_offsets_s = torch.tensor([0, s_cond.sum()], device=voxel_coords.device, dtype=torch.int)
            query_locs1, query_feats1, query_boxes1, query_inds1 = self.point_aggregator1(
                voxel_coords_float_[s_cond],
                voxel_output_feats_[s_cond],
                voxel_box_preds_[s_cond],
                batch_offsets=voxel_batch_offsets_s,
                batch_size=batch_size,
                sampled_before=False,
            )

        # NOTE iterative sampling
        cls_logits_final = []
        mask_logits_final = []
        conf_logits_final = []
        box_preds_final = []
        mask_pure_ray = []
        mask_angles = []
        if self.is_mask:
            union_mask = torch.zeros_like(query_inds1[0])

        if self.iterative_sampling:
            n_sample = min(max(150, int(len(object_idxs) / 300)), 500)
            n_sample_arr = [
                int(-(n_sample / 2) * math.log(0.1)),
                int(-(n_sample / 2) * math.log(0.2)),
                int(-(n_sample / 2) * math.log(0.4)),
            ]
        else:
            n_sample_arr = [256]

        query_locs2 = query_locs1.clone()
        query_feats2 = query_feats1.clone()
        query_boxes2 = query_boxes1.clone()
        
        for i in range(len(n_sample_arr)):
            self.point_aggregator2.n_sample = min(n_sample_arr[i], query_locs2.shape[1])
            query_locs, query_feats, query_boxes, query_inds = self.point_aggregator2(
                query_locs2, query_feats2, query_boxes2, batch_size, sampled_before=False
            )
           
            if self.is_mask:
                forward_head_outputs = self.forward_head(
                    query_feats,
                    query_locs,
                    dc_mask_features,
                    dc_coords_float,
                    dc_box_preds,
                    dc_batch_offsets,
                    inference=True,
                )
            else:
                forward_head_outputs = self.forward_head(
                    query_feats,
                    query_locs,
                    [],
                    dc_coords_float,
                    dc_box_preds,
                    dc_batch_offsets,
                    inference=True,
                    instance_sphere=instance_sphere,
                    instance_label=instance_cls
                )
            
            if forward_head_outputs is None:
                break
            if self.is_mask:
                cls_logits, mask_logits, conf_logits, box_preds = forward_head_outputs
                
                mask_logit = mask_logits[:, voxel_spps]
                mask_logits_final.append(mask_logit)
            else:
                cls_logits, conf_logits, box_preds = forward_head_outputs
            
            box_in_3D = get_3D_locs_from_rays(box_preds[:,:-3], self.angles_ref, box_preds[:,-3:])
           
            box_preds = torch.cat([box_in_3D, box_preds[:,None,-3:]],1)
            box_preds = box_preds.reshape(len(box_preds),-1)
            #box_preds.shape
            #torch.Size([28, 138])       
           
            if self.is_debug: #Hard GT assigning
                ####################debug
                #Using ray gt
              
                
                box_preds = instance_sphere.reshape(instance_sphere.shape[0],-1)

                if False:

                    for idx_debug in range(len(instance_cls)):
                        obj_cls = instance_cls[idx_debug]
                     
                        if obj_cls.item() in self.rays_anchors:
                        
                            obj_anchor = torch.tensor(self.rays_anchors[obj_cls.item()]).float().cuda()
                            
                            box_in_3D_ = get_3D_locs_from_rays(obj_anchor[None,:-3], self.angles_ref, instance_sphere[idx_debug,-1][None,:])
                            box_preds_debug = torch.cat([box_in_3D_, instance_sphere[idx_debug,-1][None,None,:]],1)
                            box_preds_debug = box_preds_debug.reshape(len(box_preds_debug),-1)
                            box_preds[idx_debug] = box_preds_debug[0]
                            
                cls_logits = torch.zeros(len(instance_sphere),cls_logits.shape[1]).float().cuda()
                conf_logits = torch.zeros(len(instance_sphere)).float().cuda()
                labels_form = torch.cat([torch.arange(len(instance_cls))[:,None].float().cuda(), instance_cls[:,None]],1)
                labels_mask = labels_form[:,1] == -100
                labels_form[labels_mask,1] = 18
                
             
                cls_logits[:]= -3
                cls_logits[labels_form[:,0].long(), labels_form[:,1].long()] = 5
                conf_logits[:] = 0.99
               

               
                #Using angular gt
                voxel_angular_offset = pointwise_angular_labels
                cls_logits_final.append(cls_logits)
                conf_logits_final.append(conf_logits)
                ####################has to be removed
                
                #box_preds_3form, cls_logits_, conf_logits_ = analyze_pred_ray(box_preds.reshape(box_preds.shape[0], (self.rays_width*self.rays_height)+1, 3), cls_logits, conf_logits, instance_sphere, instance_cls,
                #                                                            center_refine=False, ray_refine=True, cls_refine=True)
                # 
                
                #box_preds = box_preds_3form.reshape(box_preds_3form.shape[0], -1)
                #cls_logits_final.append(cls_logits_)
                #conf_logits_final.append(conf_logits_)
                ###

            else:

                cls_logits_final.append(cls_logits)
                conf_logits_final.append(conf_logits)
            box_preds_final.append(box_preds)
                ###
            
            
            masks_ray_, masks_angles_ = get_mask_from_polar_single(voxel_coords_float, box_preds, self.rays_width, self.rays_height, 
                                                                   is_angular=True, pointwise_angular=voxel_angular_offset,
                                                                    sim_ths = 0.90 )
            
            self.instance_labels = instance_labels
            max_gt2pred, iou_gt2pred = manual_iou(masks_ray_, instance_labels, v2p_map, p2v_map[:,1], p2v=True)
            
            #print(max_gt2pred)
            ####For debugging why the eval result is bad while IoU is good
            
            #best_indice = torch.argmax(iou_gt2pred,1)
            
            #box_preds_final[0] = box_preds_final[0][best_indice]
            #cls_logits_final[0] = cls_logits_final[0][best_indice]
            #conf_logits_final[0] = conf_logits_final[0][best_indice]
            #masks_ray_ = masks_ray_[best_indice]
            #masks_angles_ = masks_angles_[best_indice]
            #### Debugging code ends here

            mask_pure_ray.append(masks_ray_)
            mask_angles.append(masks_angles_)


            if i == len(n_sample_arr) - 1:
                break
            
            if self.is_mask:
                pdb.set_trace()
                mask_logits_0 = (mask_logits[:, voxel_spps_][:, query_inds1[0].long()].detach() > 0).float()
                union_mask = union_mask + mask_logits_0.sum(dim=0)

            nonvisited_query_inds1 = torch.nonzero(union_mask == 0).view(-1)
            if len(nonvisited_query_inds1) == 0:
                break

            query_locs2 = torch.gather(
                query_locs1, dim=1, index=nonvisited_query_inds1[None, :, None].expand(-1, -1, query_locs1.shape[-1])
            )
            query_feats2 = torch.gather(
                query_feats1, dim=2, index=nonvisited_query_inds1[None, None, :].expand(-1, query_feats1.shape[1], -1)
            )
            query_boxes2 = torch.gather(
                query_boxes1, dim=1, index=nonvisited_query_inds1[None, :, None].expand(-1, -1, query_boxes1.shape[-1])
            )
        '''
        if len(cls_logits_final) == 0:
            ret.update(dict(pred_instances=[]))
            return ret
        
        mask_ray_final = torch.cat(mask_pure_ray)
        mask_angles_final = torch.cat(mask_angles)
        cls_logits_final = torch.cat(cls_logits_final)
        if self.is_mask:
            mask_logits_final = torch.cat(mask_logits_final)
        conf_logits_final = torch.cat(conf_logits_final)
        box_preds_final = torch.cat(box_preds_final)
        #pdb.set_trace()
        '''
        pred_instances = self.get_instance(
            scan_ids[0],
            cls_logits_final,
            mask_logits_final,
            conf_logits_final,
            box_preds_final,
            v2p_map,
            voxel_semantic_scores_pred,
            spps,
            logit_thresh=self.test_cfg.logit_thresh,
            score_thresh=self.test_cfg.score_thresh,
            npoint_thresh=self.test_cfg.npoint_thresh,
        )
        #pdb.set_trace()
        '''
        
        pred_instances = self.get_instance(
            scan_ids[0],
            cls_logits_final,
            #mask_angles_final,
            mask_ray_final,
            conf_logits_final,
            box_preds_final,
            v2p_map,
            voxel_semantic_scores_pred,
            spps,
            logit_thresh=self.test_cfg.logit_thresh,
            score_thresh=self.test_cfg.score_thresh,
            npoint_thresh=self.test_cfg.npoint_thresh,
            instance_labels=instance_sphere
        )
        
        ret.update(dict(pred_instances=pred_instances))
        return ret

    def forward_backbone(
        self, input_feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map, x4_split=False
    ):

        context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with context():
            if self.with_coords:
                input_feats = torch.cat((input_feats, coords_float), 1)

            voxel_input_feats = voxelization(input_feats, p2v_map)
            input = spconv.SparseConvTensor(voxel_input_feats, voxel_coords.int(), spatial_shape, batch_size)

            if x4_split:
                output_feats, output_batch_idxs = self.forward_4_parts(input)
            else:

                output = self.input_conv(input)
                output = self.unet(output)
                output = self.output_layer(output)

                output_feats = output.features
                output_batch_idxs = output.indices[:, 0]

            return output_feats, output_batch_idxs

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]

        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    def forward_4_parts(self, x):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1
            )

            output = self.input_conv(x_new)
            output = self.unet(output)
            output = self.output_layer(output)

            outs.append(output.features)
        outs = torch.cat(outs, dim=0)
        idxs = torch.zeros((outs.shape[0]), dtype=torch.int, device=outs.device)
        return outs, idxs

    def forward_pointwise_head(self, output_feats):

        #context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        #with context():
        
        #semantic_scores = self.semantic_linear(output_feats[:,:,None]).squeeze(-1)
        semantic_scores = self.semantic_linear(output_feats)
        centroid_offset = self.offset_linear(output_feats[:,:,None]).squeeze(-1)
        angular_offset = self.angular_linear(output_feats[:,:,None]).squeeze(-1)
        corners_offset = self.offset_vertices_linear(output_feats[:,:,None]).squeeze(-1)
        
       
       
        box_conf = self.box_conf_linear(output_feats[:,:,None])[:,0,0]#.squeeze()
        return semantic_scores, centroid_offset, corners_offset, angular_offset, box_conf

    def spp_pool(self, voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets):
        spp_batch_offsets = [0]
        for b in range(len(voxel_batch_offsets) - 1):
            b_s, b_e = voxel_batch_offsets[b], voxel_batch_offsets[b + 1]
            n_spps_ = len(torch.unique(voxel_spps[b_s:b_e]))

            spp_batch_offsets.append(spp_batch_offsets[-1] + n_spps_)
        spp_batch_offsets = torch.tensor(spp_batch_offsets, dtype=torch.long, device=voxel_batch_offsets.device)

        spp_coords_float = custom_scatter_mean(voxel_coords_float, voxel_spps, pool=self.use_spp_pool)

        
        spp_box_preds = custom_scatter_mean(voxel_box_preds, voxel_spps, pool=self.use_spp_pool)
        spp_output_feats = custom_scatter_mean(voxel_output_feats, voxel_spps, pool=self.use_spp_pool)

        return spp_coords_float, spp_output_feats, spp_box_preds, spp_batch_offsets

    def forward_head(
        self, queries_features, queries_locs, mask_features_, locs_float_, box_preds_, batch_offsets_, inference=False,
        instance_sphere=None, instance_label=None
    ):

        queries_features = self.inst_shared_mlp(queries_features)

        cls_logits = self.inst_sem_head(queries_features).transpose(1, 2)  # batch x n_queries x n_classes
        conf_logits = self.inst_conf_head(queries_features).transpose(1, 2).squeeze(-1)  # batch x n_queries
        #box_offsets_preds = self.inst_box_head(queries_features).transpose(1, 2)  # batch x n_queries x 6
        
       
        box_offsets_preds = self.inst_box_head(queries_features).transpose(1, 2)
        box_offsets_center_preds = self.inst_box_center_head(queries_features).transpose(1, 2)
        
        ###
        box_offsets_preds = box_offsets_preds.view(box_offsets_preds.shape[0], box_offsets_preds.shape[1], self.instance_classes, self.rays_width*self.rays_height)
        box_offsets_center_preds = box_offsets_center_preds[:,:,None,:].repeat(1,1,self.instance_classes,1)
        ###
        
        
        #box_offsets_preds[:,:,-3:] = box_offsets_preds[:,:,-3:] + queries_locs
        #queries_box_preds = box_offsets_preds
        queries_box_preds = box_offsets_center_preds + queries_locs[:,:,None,:]
        queries_box_preds = torch.cat([box_offsets_preds, queries_box_preds],3)
        
         
        '''
        if self.training == False or inference: 
            #queries_box_preds: torch.Size([12, 256, 48])
            #self.rays_anchors_tensor: torch.Size([19, 45])
            #pred_classes: torch.Size([12, 256])
            _, pred_classes = torch.max(cls_logits,2)
            norm_params = self.rays_norm_params[pred_classes]
            queries_box_preds[:,:,:-3] = ((queries_box_preds[:,:,:-3])*norm_params[:,:,:,1]) + norm_params[:,:,:,0]
            queries_box_preds[:,:,:-3] += self.rays_anchors_tensor[pred_classes]
        '''
        if inference:
            
            cls_preds = torch.argmax(cls_logits[0], dim=-1)
            fg_conds = cls_preds < self.instance_classes
            cls_preds = cls_preds[fg_conds]
            if False and self.is_debug:
                dist_with_gt = torch.cdist(queries_box_preds[0], instance_sphere.reshape(instance_sphere.shape[0],-1))
                _, best_indice = torch.min(dist_with_gt,1)
                
                if True: ### only center
                    queries_box_preds = queries_box_preds.reshape(queries_box_preds.shape[0], queries_box_preds.shape[1], (self.rays_width*self.rays_height)+3)
                    queries_box_preds[:,best_indice,-1] = instance_sphere[best_indice][:,-1]
                    queries_box_preds = queries_box_preds.view(box_offsets_preds.shape[0],box_offsets_preds.shape[1], box_offsets_preds.shape[2]*box_offsets_preds.shape[3])
                if False: ### entire ray parameters
                    queries_box_preds = instance_sphere[best_indice]
                    queries_box_preds = queries_box_preds.reshape(1,queries_box_preds.shape[0],-1)
                
                #box_preds = box_preds.reshape(box_preds.shape[0],-1)
                box_labels = instance_label[best_indice] 
                box_labels[box_labels<0] = 18
                #pdb.set_trace()
                ious, _ = batch_giou_cross_polar(queries_box_preds[0],instance_sphere.reshape(instance_sphere.shape[0],-1))
                ious_conf= (torch.mean(ious,1) + torch.amax(ious,1))/2.0
                
                
                
                box_labels_form = torch.zeros(1,len(box_labels),19).float().cuda()
                box_labels_form[0,:,:] = -3
                labels_form = torch.cat([torch.arange(len(box_labels))[:,None].cuda(), box_labels[:,None]],1)
                box_labels_form[0,labels_form[:,0],labels_form[:,1]] = 5
            
                cls_logits = box_labels_form
                conf_logits = ious_conf[None,:]


                cls_preds = torch.argmax(box_labels_form[0], dim=-1)
                fg_conds = cls_preds < self.instance_classes
            elif fg_conds.sum() == 0:
                

                return None
            
            queries_features = queries_features[0, :, fg_conds].unsqueeze(0)
            cls_logits = cls_logits[0, fg_conds].unsqueeze(0)
            conf_logits = conf_logits[0, fg_conds].unsqueeze(0)
            
            queries_box_preds = queries_box_preds[0, fg_conds].unsqueeze(0)
            cls_ext_mat = torch.cat([torch.arange(len(cls_preds))[:,None].long().cuda(), cls_preds[:,None]],1)
            queries_box_preds = queries_box_preds[:,cls_ext_mat[:,0], cls_ext_mat[:,1]]

            queries_locs = queries_locs[0, fg_conds].unsqueeze(0)

        batch_size = queries_features.shape[0]
        n_queries = queries_features.shape[2]

        
        if self.is_mask:
            controllers = (
                self.inst_mask_head(queries_features.permute(0, 2, 1)[..., None].flatten(0, 1))
                .squeeze(dim=2)
                .reshape(batch_size, n_queries, -1)
            )
            mask_logits = []
            for b in range(batch_size):
                start, end = batch_offsets_[b], batch_offsets_[b + 1]
                num_points = end - start
                if num_points == 0:
                    mask_logits.append(None)
                    continue

                controller = controllers[b]  # n_queries x channel
                
                weights, biases = self.parse_dynamic_params(controller, self.mask_dim_out)
                
                mask_feature_b = mask_features_[start:end]
                locs_float_b = locs_float_[start:end]
                box_preds_b = box_preds_[start:end]
                queries_locs_b = queries_locs[b]
                queries_box_preds_b = queries_box_preds[b]

                # NOTE as # of points in s3dis is very big, we split dyco into multiple chunks to avoid OOM
                if not self.training and self.dataset_name == "s3dis":
                    num_chunks = 16
                    chunk_size = math.ceil(n_queries / num_chunks)
                else:
                    num_chunks = 1
                    chunk_size = n_queries

                mask_logit = []
                for i in range(num_chunks):
                    chunk_start = i * chunk_size
                    chunk_end = (i + 1) * chunk_size
                    weights_split = [w[chunk_start:chunk_end] for w in weights]
                    biases_split = [b[chunk_start:chunk_end] for b in biases]
                    
                    mask_logit_ = self.mask_heads_forward(
                        mask_feature_b,
                        weights_split,
                        biases_split,
                        weights_split[0].shape[0],
                        locs_float_b,
                        box_preds_b,
                        queries_locs_b[chunk_start:chunk_end],
                        queries_box_preds_b[chunk_start:chunk_end],
                    )
                
                    mask_logit.append(mask_logit_)
                mask_logit = torch.cat(mask_logit, dim=0)

                mask_logits.append(mask_logit)
        if self.is_mask:
            if inference:
                return cls_logits[0], mask_logits[0], conf_logits[0], queries_box_preds[0]
            return cls_logits, mask_logits, conf_logits, queries_box_preds
        if inference:
                return cls_logits[0], conf_logits[0], queries_box_preds[0]
        return cls_logits, conf_logits, queries_box_preds

    def parse_dynamic_params(self, params, out_channels):
        assert params.dim() == 2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        #weight_splits[0] = weight_splits[0].reshape(num_instances, out_channels + 6, out_channels)
        
        weight_splits[0] = weight_splits[0].reshape(num_instances, out_channels + self.rays_width*(self.rays_height)*3 +3 + 3, out_channels)
        bias_splits[0] = bias_splits[0].reshape(num_instances, out_channels)
        weight_splits[1] = weight_splits[1].reshape(num_instances, out_channels, 1)
        bias_splits[1] = bias_splits[1].reshape(num_instances, 1)

        return weight_splits, bias_splits  # LIST OF [n_queries, C_in, C_out]

    def mask_heads_forward(
        self, mask_features, weights, biases, num_insts, coords_, boxes_, queries_coords, queries_boxes
    ):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        x = mask_features.permute(2, 1, 0).repeat(num_insts, 1, 1)  # num_inst * c * N_mask

        relative_coords = queries_coords.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3)  # N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0, 2, 1)

        #queries_boxes_dim = queries_boxes[:, 3:] - queries_boxes[:, :3]
        #boxes_dim = boxes_[:, 3:] - boxes_[:, :3]
        #relative_boxes = torch.abs(
        #    queries_boxes_dim.reshape(-1, 1, 3) - boxes_dim.reshape(1, -1, 3)
        #)  # N_inst * N_mask * 3

        queries_boxes_dim = queries_boxes
        boxes_dim = boxes_
        
        relative_boxes = torch.abs(
            queries_boxes_dim.reshape(-1, 1, self.rays_width*(self.rays_height)*3 + 3) - boxes_dim.reshape(1, -1, self.rays_width*(self.rays_height)*3 + 3)
        )  # N_inst * N_mask * 3
        
        relative_boxes = relative_boxes.permute(0, 2, 1)

        x = torch.cat([relative_coords, relative_boxes, x], dim=1)  # num_inst * (3+c) * N_mask
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            if i < n_layers - 1:
                x = torch.einsum("qab,qan->qbn", w, x) + b.unsqueeze(-1)
                x = F.relu(x)
            else:
                x = torch.einsum("qab,qan->qbn", w, x) # NOTE Empirically, we do not add biases in last dynamic_conv layer
                x = x.squeeze(1)
        
        return x

    def get_instance(
        self,
        scan_id,
        cls_logits,
        mask_logits,
        conf_logits,
        box_preds,
        v2p_map,
        semantic_scores_preds,
        spps,
        logit_thresh=0.0,
        score_thresh=0.1,
        npoint_thresh=100,
        instance_labels=None
    ):

        # NOTE only batch 1 when test

        instances = []
        
        # NOTE for S3DIS: it requires instance mask of background classes (floor, ceil)
        for i in self.sem2ins_classes:
            
            mask_pred = semantic_scores_preds == i

            mask_pred = mask_pred[v2p_map.long()]

            if torch.count_nonzero(spps == self.ignore_label) != spps.shape[0]:
                mask_pred = superpoint_align(spps, mask_pred.unsqueeze(0)).squeeze(0)

            pred = {}
            pred["scan_id"] = scan_id
            pred["label_id"] = i + 1
            pred["conf"] = 1.0
            
            # rle encode mask to save memory
            pred["pred_mask"] = rle_encode_gpu(mask_pred)
            instances.append(pred)
        
        cls_logits = F.softmax(cls_logits, dim=-1)
        cls_pred = torch.argmax(cls_logits, dim=-1)  # n_mask
        conf_logits = torch.clamp(conf_logits, 0.0, 1.0)
        masks_pred = mask_logits >= logit_thresh
        
        cls_logits_scores = torch.gather(cls_logits, 1, cls_pred.unsqueeze(-1)).squeeze(-1)
        scores = torch.sqrt(conf_logits * cls_logits_scores)

        scores_cond = (conf_logits > score_thresh) & (cls_logits_scores > score_thresh)
        
        cls_final = cls_pred[scores_cond]
        masks_final = masks_pred[scores_cond]
        scores_final = scores[scores_cond]
        boxes_final = box_preds[scores_cond]

        proposals_npoints = torch.sum(masks_final, dim=1)
        npoints_cond = proposals_npoints >= npoint_thresh
        cls_final = cls_final[npoints_cond]
        masks_final = masks_final[npoints_cond]
        scores_final = scores_final[npoints_cond]
        boxes_final = boxes_final[npoints_cond]
        
        # NOTE NMS
        #####debug
        '''
        if self.is_debug == False:
            max_gt2pred, iou_gt2pred = manual_iou(masks_final, self.instance_labels, v2p_map, self.p2v_map[:,1], p2v=True)
            best_indice = torch.argmax(iou_gt2pred,1)
            masks_final = masks_final[best_indice]
            cls_final = cls_final[best_indice]
            scores_final = scores_final[best_indice]
            boxes_final = boxes_final[best_indice]
            cls_final = self.instance_cls.clone()
            cls_final[cls_final == -100] = 18
            print(max_gt2pred)
        '''
        #####debug


        masks_final, cls_final, scores_final, boxes_final = nms(masks_final, cls_final, scores_final, boxes_final, test_cfg=self.test_cfg )
       
        if len(cls_final) == 0:
            return instances

        # NOTE devoxelization
        
        masks_final = masks_final[:, v2p_map.long()]

        # NOTE due to the quantization error, we apply superpoint refinement to the final mask after devoxelization
        
        spp_masks_final = custom_scatter_mean(
            masks_final,
            spps[None, :].expand(len(masks_final), -1),
            dim=-1,
            pool=self.use_spp_pool,
            output_type=torch.float32,
        )
       
        masks_final = (spp_masks_final >= 0.5)[:, spps]

        valid_masks_final_after_spp = torch.sum(masks_final, dim=1) >= npoint_thresh
        masks_final = masks_final[valid_masks_final_after_spp]
        scores_final = scores_final[valid_masks_final_after_spp]
        cls_final = cls_final[valid_masks_final_after_spp]
        boxes_final = boxes_final[valid_masks_final_after_spp]

        # NOTE save results
        scores_final = scores_final.cpu().numpy()
        cls_final = cls_final.cpu().numpy()
        boxes_final = boxes_final.cpu().numpy()

        # NOTE rle encode mask to save memory
        rle_masks = rle_encode_gpu_batch(masks_final)

        for i in range(cls_final.shape[0]):
            pred = {}
            pred["scan_id"] = scan_id

            if self.dataset_name == "scannetv2":
                pred["label_id"] = cls_final[i] + 1
            elif self.dataset_name == "scannet200":
                pred["label_id"] = cls_final[i] + 1
            elif self.dataset_name == "s3dis":
                pred["label_id"] = cls_final[i] + 3
            elif self.dataset_name == "stpls3d":
                pred["label_id"] = cls_final[i] + 1
            else:
                raise Exception("Invalid dataset name")

            pred["conf"] = scores_final[i]

            pred["pred_mask"] = rle_masks[i]

            instances.append(pred)

        return instances
