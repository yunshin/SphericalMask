import numpy as np
import torch
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
import argparse
import datetime
import os
import os.path as osp
import shutil
import time
from isbnet.data import build_dataloader, build_dataset
from isbnet.evaluation import PointWiseEval, ScanNetEval
from isbnet.model import ISBNet
from isbnet.model import PolarNet
from isbnet.model import PolarNet_Direct
from isbnet.model import PolarNet_Direct_EXP3
from isbnet.model import PolarNet_Dyco_Ray_Cluster
from isbnet.model.criterion import Criterion
from isbnet.model.criterion_polar import Criterion_Polar
from isbnet.util import (
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    checkpoint_save,
    cosine_lr_after_step,
    get_dist_info,
    get_max_memory,
    get_root_logger,
    init_dist,
    is_main_process,
    is_multiple,
    is_power2,
    load_checkpoint,
)
from isbnet.model.model_utils import (get_instance_info_polarnet, batch_giou_cross_polar, get_3D_locs_from_rays, batch_giou_corres_polar)
from scipy.stats import norm
from isbnet.model.criterion_dyco_ray_cluster import Criterion_Dyco_Ray_Cluster
from sklearn.decomposition import PCA
import pickle
import pdb

np.random.seed(0)
torch.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--only_backbone", action="store_true", help="only train backbone")
    parser.add_argument("--trainall", action="store_true", help="only train backbone")
    args = parser.parse_args()
    return args



def analyize(epoch, model, optimizer, val_loader, cfg, logger, writer):
    logger.info("Validation")
    all_pred_insts, all_sem_labels, all_ins_labels = [], [], []

    val_set = val_loader.dataset

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(val_set.CLASSES, dataset_name=cfg.data.train.type)

    torch.cuda.empty_cache()

    # FIXME Only during training, to avoid oom, we set iterative_sampling = False
    model.iterative_sampling = False
    model.is_analyze = True

    cls_dict = {}

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            print('{0}/{1}'.format(i,len(val_loader)))
            batch['type'] = 1
            # NOTE Skip large scene of s3dis during traing to avoid oom
            if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 3000000:
                continue
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)
            
            instance_sphere = res[2].cpu().numpy()
            instance_center = instance_sphere[:,-1]
            rays_gt_list = res[-1].cpu().numpy()
            obj_classes = res[0].cpu().numpy()
            
            #rays_gt_with_center = np.concatenate([rays_gt_list[:,:,0], instance_center],1)
            
            rays_gt_with_center = rays_gt_list[:,:,0]
            
            for idx in range(len(obj_classes)):

                cls_num = obj_classes[idx]
                if cls_num < 0:
                    continue
                if cls_num not in cls_dict:
                    cls_dict[cls_num] = []

                cls_dict[cls_num].append(rays_gt_with_center[idx])

    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_stats_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
      pickle.dump(cls_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #pdb.set_trace()
    #pdb.set_trace()
    print('saving done.')

def get_pcd_anchor(rays, remove_zeros=False):
    if remove_zeros:
       
        rays_mean, rays_anchor, anchored_std, rays_std = [],[],[],[]
        for idx in range(rays.shape[1]):

            rays_in_angle = rays[:,idx]
            zero_mask = rays_in_angle > 1e-6
            rays_in_angle = rays_in_angle[zero_mask]
            
            mean, std = np.mean(rays_in_angle), np.std(rays_in_angle)
            rays_in_angle_anchord = rays_in_angle - mean
        
            std_anchord = np.std(rays_in_angle_anchord)
            rays_mean.append(mean);rays_std.append(std);anchored_std.append(std_anchord)
        return np.array(rays_mean), np.array(rays_std), np.array(anchored_std)


        rays_mean = np.mean(rays,0)
        rays_anchored = rays - rays_mean

        anchored_std = np.std(rays_anchored,0)
        rays_std = np.std(rays,0)

    pca = PCA(n_components=5)
    rays_transposed = rays.T
    pca.fit(rays_transposed)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    pca.transform(rays_transposed)
    pdb.set_trace()
    rays_mean = np.mean(rays,0)
    rays_anchored = rays - rays_mean

    anchored_std = np.std(rays_anchored,0)
    rays_std = np.std(rays,0)

    return rays_mean, rays_std, anchored_std

def get_range_anchor(rays, remove_zeros=False, num_range=5):
    if remove_zeros:
       
        rays_mean, rays_anchor, anchored_std, rays_std = [],[],[],[]
        for idx in range(rays.shape[1] - 3):

            rays_in_angle = rays[:,idx]
            zero_mask = rays_in_angle > 1e-2
            rays_in_angle = rays_in_angle[zero_mask]
            mean, std = np.mean(rays_in_angle), np.std(rays_in_angle)
            if len(rays_in_angle) == 0:
                pdb.set_trace()
            min_val, max_val = np.amin(rays_in_angle), np.amax(rays_in_angle)
            step_size = (max_val - min_val) / num_range
            range_vals = np.arange(min_val, max_val+0.01, step_size)
            if range_vals[-1] < max_val:
                range_vals[-1] = max_val + 0.01
            range_vals += 0.02
            #range_vals[1] += 0.02
            
            rays_in_angle_anchord = rays_in_angle - mean
        
            std_anchord = np.std(rays_in_angle_anchord)
            rays_mean.append(range_vals);rays_std.append(std);anchored_std.append(std_anchord)
        
        return np.array(rays_mean), np.array(rays_std), np.array(anchored_std)


        rays_mean = np.mean(rays,0)
        rays_anchored = rays - rays_mean

        anchored_std = np.std(rays_anchored,0)
        rays_std = np.std(rays,0)
    
    rays_mean = np.mean(rays,0)
    rays_anchored = rays - rays_mean

    anchored_std = np.std(rays_anchored,0)
    rays_std = np.std(rays,0)

    return rays_mean, rays_std, anchored_std

def get_avg_anchor(rays, remove_zeros=False):
    if remove_zeros:
       
        rays_mean, rays_anchor, anchored_std, rays_std = [],[],[],[]
        for idx in range(rays.shape[1]):

            rays_in_angle = rays[:,idx]
            zero_mask = rays_in_angle > 1e-6
            rays_in_angle = rays_in_angle[zero_mask]
            mean, std = np.mean(rays_in_angle), np.std(rays_in_angle)
            rays_in_angle_anchord = rays_in_angle - mean
        
            std_anchord = np.std(rays_in_angle_anchord)
            rays_mean.append(mean);rays_std.append(std);anchored_std.append(std_anchord)
        return np.array(rays_mean), np.array(rays_std), np.array(anchored_std)


        rays_mean = np.mean(rays,0)
        rays_anchored = rays - rays_mean

        anchored_std = np.std(rays_anchored,0)
        rays_std = np.std(rays,0)
    
    rays_mean = np.mean(rays,0)
    rays_anchored = rays - rays_mean

    anchored_std = np.std(rays_anchored,0)
    rays_std = np.std(rays,0)

    return rays_mean, rays_std, anchored_std
def cdf_sequential(data,mean,std):

    result = np.zeros((len(mean), len(data)))

    for mean_idx in range(len(mean)):

        
        
            
        value = norm.sf(x=data[:, mean_idx], loc=mean[mean_idx], scale=std[mean_idx])
        
        
        inverse_value = norm.ppf(value  , loc=-mean[mean_idx], scale=std[mean_idx])
        if abs((-1*inverse_value)-data[:, mean_idx]).max() > 0.0001 :
            pdb.set_trace()

        result[mean_idx, :] = value
    return result.mean(1), result.std(1)
def analyze_by_loading(path,cfg):
    with open(path, 'rb') as f:
      rays_dict_by_class = pickle.load(f)


    save_anchor_dict = {}
    save_anchor_norm_dict = {}
    save_anchor_max_dict = {}
    save_anchor_cdf_dict = {}
    save_anchor_range_dict = {}
    for key in rays_dict_by_class.keys():
        cls_num = key

        rays = np.array(rays_dict_by_class[cls_num])
        rays_max = np.amax(rays,0)
        save_anchor_max_dict[cls_num] = rays_max
        
        print(key)
        avg_anchor, ori_std, anchored_std = get_avg_anchor(rays)
        avg_anchor_wo_zero, ori_std_wo_zero, anchored_std_wo_zero = get_avg_anchor(rays, remove_zeros=True)
        #range_anchor_wo_zero, ori_std_wo_zero, anchored_std_wo_zero = get_range_anchor(rays, remove_zeros=True)
        #pcd_anchor_w_zero, pcd_std_w_zero, pcd_std_wo_zero = get_pcd_anchor(rays, remove_zeros=False)
        
        #ta = rays[:,:-3]
        #tt = abs(ta[:,0] - 1.8121)
        #if tt.min() < 1e-2:
        #    pdb.set_trace()
        rays_min = np.amin(rays,0)
        
        rays_norm = (rays - rays_min)/(rays_max - rays_min)

        #prob = normpdf(rays[0,:-3] ,avg_anchor[:-3],ori_std[:-3])

        #prob = normpdf(rays[0,:-3] ,avg_anchor_wo_zero[:-3], ori_std_wo_zero[:-3])
        
        #mean, std = cdf_sequential(rays, avg_anchor[:-3], ori_std[:-3])
        
       
        avg_anchor_norm, ori_std_norm, anchored_std_norm = get_avg_anchor(rays_norm)
        avg_anchor_norm_, ori_std_norm_, anchored_std_norm_ = get_avg_anchor(rays_norm)
        
        save_anchor_cdf_dict[cls_num] = [avg_anchor[:-3], ori_std[:-3]]
        save_anchor_norm_dict[cls_num] = [rays_min, rays_max - rays_min]
        save_anchor_dict[cls_num] = avg_anchor_wo_zero
        
        #save_anchor_range_dict[cls_num] = range_anchor_wo_zero
        #save_anchor_dict[cls_num] = avg_anchor
    
    #with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_wo_zero.pkl', 'wb') as handle:
   
    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_cdf_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
        pickle.dump(save_anchor_cdf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
        pickle.dump(save_anchor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_max_vals_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
        pickle.dump(save_anchor_max_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_wo_zero_norm_params_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
        pickle.dump(save_anchor_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('/root/data/pcdet_data/scannet_pcdet_data/rays_range_anchors_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), 'wb') as handle:
    #    pickle.dump(save_anchor_range_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saving done in loading.')
    pdb.set_trace()
    
def optimize_anchors(path, ref_angles):

    with open(path, 'rb') as f:
      rays_dict_by_class = pickle.load(f)

    num_cls = len(list(rays_dict_by_class.keys()))
    gt_dict_3D_pos = {}
    anchors_params_list = []
    for key in rays_dict_by_class.keys():
        cls_num = key
        rays = np.array(rays_dict_by_class[cls_num])
        
        ray_centers = rays[:,-3:]
        rays = rays[:,:-3]
        avg_anchor_wo_zero, ori_std_wo_zero, anchored_std_wo_zero = get_avg_anchor(rays, remove_zeros=True)
        #gts_in_3D = get_3D_locs_from_rays(torch.tensor(rays), torch.tensor(ref_angles), torch.tensor(ray_centers))
        gt_dict_3D_pos[cls_num] = rays
        anchors = torch.ones((1,45)).float().cuda()
        anchors[0] = torch.tensor(avg_anchor_wo_zero).float().cuda()
        
        anchors_params_list.append(torch.tensor(anchors, requires_grad=True))

    
    
    #anchors = torch.tensor(anchors,requires_grad=True)
    optimizer = torch.optim.Adam(anchors_params_list, lr=0.001)
    num_iter = 200
    
    for iteration in range(num_iter):

        cnt_row = 0
        iou_loss = 0
        var_dict = {}
        for key in gt_dict_3D_pos:
            
            anchors_to_optimize = anchors_params_list[cnt_row]
            cnt_row +=1
            rays = torch.tensor(np.array(rays_dict_by_class[key])).float().cuda()
            ray_centers = rays[:,-3:]
            rays = rays[:,:-3]
            #anchors_in_3D = get_3D_locs_from_rays(anchors_to_optimize.repeat(len(ray_centers),1), torch.tensor(ref_angles).float().cuda(), torch.tensor(ray_centers).float().cuda())
            
            loss_total = 0
            for idx_ray in range(rays.shape[1]):
                
                non_zero_mask = rays[idx_ray] > 1e-6 
                ray_to_reach = rays[idx_ray,non_zero_mask]      
                
                diff = torch.abs(ray_to_reach[:,None] - anchors_to_optimize[:,idx_ray])
                loss = diff.sum() / len(diff)
                loss_total += loss
            optimizer.zero_grad()
            loss_total.backward()
            
            optimizer.step()
            
            var_dict[key] = loss.item()
        if iteration % 10 == 0:
            print('Iter {0}: {1}'.format(iteration, var_dict))
    save_anchor_dict = {}
    cnt = 0
    for key in gt_dict_3D_pos:
        save_anchor_dict[key] = anchors_params_list[cnt][0].detach().cpu().numpy()
        cnt += 1
    
    pdb.set_trace()
    with open('/root/data/pcdet_data/scannet_pcdet_data/rays_anchors_optim.pkl', 'wb') as handle:
        pickle.dump(save_anchor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done')
   
def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist
    
    # work_dir & logger
    '''
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        dataset_name = cfg.data.train.type
        cfg.work_dir = osp.join("./work_dirs", dataset_name, osp.splitext(osp.basename(args.config))[0], args.exp_name)
    
    cfg.work_dir = Path('/root/data/pcdet_models/ISB_model/with_angles')
    '''
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    if args.only_backbone:
        logger.info("Only train backbone")
        cfg.model.semantic_only = True
        cfg.model.fixed_modules = []

    if args.trainall:
        logger.info("Train all !!!!!!!!!!!!!!!!")
        cfg.model.semantic_only = False
        cfg.model.fixed_modules = []

    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    logger.info(f"Mix precision training: {cfg.fp16}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    logger.info(f"Save at: {cfg.work_dir}")

    # model
  
    if False and 'polar' in cfg.model.model_name:
        criterion = Criterion_Polar(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster': 
        criterion = Criterion_Dyco_Ray_Cluster(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    else:
        criterion = Criterion(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    
    if cfg.model.model_name == 'isbnet':
        model = ISBNet(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster':
        
        model = PolarNet_Dyco_Ray_Cluster(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    else:
        model = PolarNet_Direct_EXP3(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    
   
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")

    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=(trainable_params < total_params)
        )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)

    train_loader = build_dataloader(train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(train_set, training=False, dist=False, **cfg.dataloader.test)

    # optim
    default_lr = cfg.optimizer.lr  # default for batch 16
    _, world_size = get_dist_info()
    total_batch_size = cfg.dataloader.train.batch_size * world_size
    scaled_lr = default_lr * (total_batch_size / 16)
    cfg.optimizer.lr = scaled_lr
    logger.info(f"Scale LR from {default_lr} (batch size 16) to {scaled_lr} (batch size {total_batch_size})")
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resume from {args.resume}")
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f"Load pretrain from {cfg.pretrain}")
        load_checkpoint(cfg.pretrain, logger, model)

    scheduler = None
    global best_metric
    best_metric = 0

    # if is_main_process():
    #     validate(0, model, optimizer, val_loader, cfg, logger, writer)

    # train and val
    logger.info("Training")
    epoch = 0
   
    angles_ref = model.angles_ref
    #analyize(epoch, model, optimizer, train_loader, cfg, logger, writer)
    #pdb.set_trace()
    analyze_by_loading('/root/data/pcdet_data/scannet_pcdet_data/rays_stats_{0}_{1}.pkl'.format(cfg.model.rays_height, cfg.model.rays_width), cfg)
    
    #optimize_anchors('/root/data/pcdet_data/scannet_pcdet_data/rays_stats.pkl', angles_ref)


if __name__ == "__main__":
    main()
