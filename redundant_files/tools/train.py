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
from isbnet.model import PolarNet_Direct_EXP1
from isbnet.model import PolarNet_Direct_EXP2
from isbnet.model import PolarNet_Direct_EXP3
from isbnet.model import PolarNet_Direct_EXP5
from isbnet.model import PolarNet_Dyco_Ray
from isbnet.model import PolarNet_Dyco_Ray_CLS
from isbnet.model import PolarNet_Dyco_Ray_Ori
from isbnet.model import PolarNet_Dyco_Ray_Cluster
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma
#from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma_Ablation
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma_Ablation_Binary
from isbnet.model import PolarNet_Dyco_Ray_Cluster_with_CLS
from isbnet.model import PolarNet_Dyco_Ray_Cluster_with_ray_CLS
from isbnet.model import SphericalMask
from isbnet.model.criterion import Criterion
from isbnet.model.criterion_polar import Criterion_Polar
from isbnet.model.criterion_polar2 import Criterion_Polar2
from isbnet.model.criterion_polar3 import Criterion_Polar3
from isbnet.model.criterion_polar4 import Criterion_Polar4
from isbnet.model.criterion_dyco_ray import Criterion_Dyco_Ray
from isbnet.model.criterion_dyco_ray_cluster import Criterion_Dyco_Ray_Cluster
from isbnet.model.criterion_dyco_ray_cluster_gamma import Criterion_Dyco_Ray_Cluster_Gamma
from isbnet.model.criterion_dyco_ray_cluster_with_cls import Criterion_Dyco_Ray_Cluster_with_CLS
from isbnet.model.criterion_dyco_ray_cluster_with_ray_cls import Criterion_Dyco_Ray_Cluster_with_ray_CLS
from isbnet.model.criterion_dyco_ray_ori_mask import Criterion_Dyco_Ray_Ori
from isbnet.model.criterion_dyco_ray_cls import Criterion_Dyco_Ray_CLS
from isbnet.model.criterion_spherical_mask import Criterion_Dyco_Ray_Cluster_Gamma_Ablation
from isbnet.model.criterion_dyco_ray_cluster_gamma_ablation_binary import Criterion_Dyco_Ray_Cluster_Gamma_Ablation_Binary
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
    load_checkpoint_encoder,
    load_checkpoint_decoder
)
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


def train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer, best_scores):

    #torch.autograd.set_detect_anomaly(True)
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)
    model.zero_grad()
    #optimizer.zero_grad()
    num_batch_accumul = 8
    loss_accumul = 0
    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
      
        start_time = time.time()
        if scheduler is None:
            cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True, epoch=epoch - 1)

        # meter_dict
        for k, v in log_vars.items():
            if k != "placeholder":
                if k not in meter_dict.keys():
                    meter_dict[k] = AverageMeter()
                meter_dict[k].update(v)

        # backward
        
        '''
        optimizer.zero_grad()
        #loss.backward()
    
        scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        
        scaler.step(optimizer)
        scaler.update()
        
        #loss = loss / num_batch_accumul
        '''
        loss_accumul += loss.item()
        
        loss = loss / num_batch_accumul
        #optimizer.zero_grad()
        #scaler.scale(loss).backward()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        if i % num_batch_accumul == 0 and i > 0:
            #loss_accumul = loss_accumul / num_batch_accumul
            optimizer.step()
            #scaler.scale(loss_accumul).backward()
            #scaler.step(optimizer)
            model.zero_grad()
            
            
            print('processed i {0} loss_accmul: {1}'.format(i,loss_accumul))
            loss_accumul = 0
            #loss_accumul = torch.zeros(1).float().cuda()
        
        #print('1 iteration time : {0}'.format(time.time() - start_time))
        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]["lr"]
       
        if scheduler is not None:
            scheduler.step()

        if is_multiple(i, 10):
        
            log_str = f"Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  "
            log_str += (
                f"lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, "
                f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"

            for key in best_scores:
                log_str += f"\n Best scores: {key}: {best_scores[key]:.4f}"

            logger.info(log_str)
        
    writer.add_scalar("train/learning_rate", lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, optimizer, val_loader, cfg, logger, writer, type):
    logger.info("Validation")
    all_pred_insts, all_sem_labels, all_ins_labels = [], [], []

    val_set = val_loader.dataset

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(val_set.CLASSES, dataset_name=cfg.data.train.type)

    torch.cuda.empty_cache()

    # FIXME Only during training, to avoid oom, we set iterative_sampling = False
    model.iterative_sampling = False
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            
            batch['type'] = type
            # NOTE Skip large scene of s3dis during traing to avoid oom
            #if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 3000000:
            #    continue
            #print(batch['scan_ids'])
            
            if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 1400000:
                print('continue')
                continue
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)
            
            
            if i % 10 == 0:
                logger.info(f"Infer scene {i+1}/{len(val_set)}")
            #pdb.set_trace()
            if cfg.model.semantic_only:
                point_eval.update(
                    res["semantic_preds"],
                    res["centroid_offset"],
                    res["corners_offset"],
                    res["semantic_labels"],
                    res["centroid_offset_labels"],
                    res["corners_offset_labels"],
                    res["instance_labels"],
                )
            else:
                all_pred_insts.append(res["pred_instances"])
                all_sem_labels.append(res["semantic_labels"])
                all_ins_labels.append(res["instance_labels"])
            #if i > 10:
            #    break            

    global best_metric

    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        miou, acc, mae = point_eval.get_eval(logger)

        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("val/Acc", acc, epoch)
        writer.add_scalar("val/Offset MAE", mae, epoch)

        if best_metric < miou:
            best_metric = miou
            checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)

    else:
        logger.info("Evaluate instance segmentation")

        # logger.info('Evaluate axis-align box prediction')
        # eval_res = scannet_eval.evaluate_box(all_pred_insts, all_gt_insts, coords)

        eval_res = scannet_eval.evaluate(all_pred_insts, all_sem_labels, all_ins_labels)
        del all_pred_insts, all_sem_labels, all_ins_labels

        writer.add_scalar("val/AP", eval_res["all_ap"], epoch)
        writer.add_scalar("val/AP_50", eval_res["all_ap_50%"], epoch)
        writer.add_scalar("val/AP_25", eval_res["all_ap_25%"], epoch)
        
        logger.info(
            "AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}".format(
                eval_res["all_ap"], eval_res["all_ap_50%"], eval_res["all_ap_25%"]
            )
        )

        if best_metric < eval_res["all_ap"]:
            best_metric = eval_res["all_ap"]
            logger.info(f"New best mAP {best_metric} at {epoch}")
            checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)
    return eval_res['all_ap']

def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
   
    if args.dist:
        init_dist()
    cfg.dist = args.dist
    
    # work_dir & logger
    
    if os.path.exists(cfg.work_dir) == False:
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
    if cfg.model.model_name == 'polarnet_direct_exp2':
        criterion = Criterion_Polar2(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_direct_exp3':
        criterion = Criterion_Polar3(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_direct_exp5':
        criterion = Criterion_Polar4(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray': 
        criterion = Criterion_Dyco_Ray(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_ori': 
        criterion = Criterion_Dyco_Ray_Ori(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_cls': 
        criterion = Criterion_Dyco_Ray_CLS(
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
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma': 
        criterion = Criterion_Dyco_Ray_Cluster_Gamma(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation' or cfg.model.model_name == 'spherical_mask': 
        criterion = Criterion_Dyco_Ray_Cluster_Gamma_Ablation(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation_binary': 
        criterion = Criterion_Dyco_Ray_Cluster_Gamma_Ablation_Binary(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_with_cls': 
        criterion = Criterion_Dyco_Ray_Cluster_with_CLS(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif 'polarnet_dyco_ray_cluster_with_ray_cls' in cfg.model.model_name:
        criterion = Criterion_Dyco_Ray_Cluster_with_ray_CLS(
            cfg.model.semantic_classes,
            cfg.model.instance_classes,
            cfg.model.semantic_weight,
            cfg.model.ignore_label,
            semantic_only=cfg.model.semantic_only,
            total_epoch=cfg.epochs,
            trainall=args.trainall,
            voxel_scale=cfg.data.train.voxel_cfg.scale,
        )
    elif 'polar' in cfg.model.model_name:
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
    elif cfg.model.model_name == 'polarnet_direct' :
        model = PolarNet_Direct(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_direct_exp1':
        model = PolarNet_Direct_EXP1(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_direct_exp2':
        model = PolarNet_Direct_EXP2(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_direct_exp3':
        model = PolarNet_Direct_EXP3(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_direct_exp5':
        model = PolarNet_Direct_EXP5(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray':
        model = PolarNet_Dyco_Ray(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cls':
        model = PolarNet_Dyco_Ray_CLS(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_ori':
        model = PolarNet_Dyco_Ray_Ori(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster':
        model = PolarNet_Dyco_Ray_Cluster(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma':
        model = PolarNet_Dyco_Ray_Cluster_Gamma(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation':
        model = PolarNet_Dyco_Ray_Cluster_Gamma_Ablation(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation_binary':
        model = PolarNet_Dyco_Ray_Cluster_Gamma_Ablation_Binary(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_with_cls':
        model = PolarNet_Dyco_Ray_Cluster_with_CLS(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_with_ray_cls':
        model = PolarNet_Dyco_Ray_Cluster_with_ray_CLS(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    elif cfg.model.model_name == 'spherical_mask':
        model = SphericalMask(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
    else:
        model = PolarNet(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=args.trainall).cuda()
   
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

    cfg.data.test['debug'] = False
    val_set = build_dataset(cfg.data.test, logger)

    train_loader = build_dataloader(train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)
   
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

    
    # train and val
    logger.info("Training")
    epoch = 0
    #pdb.set_trace()
    #ap = validate(epoch, model, optimizer, train_val_loader_vis, cfg, logger, writer, type=2)#vis
    #
    pdb.set_trace()
    ap = validate(epoch, model, optimizer, val_loader, cfg, logger, writer, type=1)#0 for ray ablation
    
    #validate(epoch, model, optimizer, val_loader, cfg, logger, writer, type=1)
    #pdb.set_trace()
    cnt_eval = 0
    best_scores = {
        'pure_ray': 0,
        'ray+exclusion_mask':0,
        'inclusion_mask':0,
        'ray+inclusion_mask':0
    }

    best_epoch = {
        'pure_ray': 0,
        'ray+exclusion_mask':0,
        'inclusion_mask':0,
        'ray+inclusion_mask':0
    }
    #torch.cuda.empty_cache()
    
    for epoch in range(start_epoch, cfg.epochs + 1):
    
        train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer, best_scores)
        #train(epoch, model, optimizer, scheduler, scaler, train_val_loader_debug, cfg, logger, writer, best_scores)
        #if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)) and is_main_process():
        if cfg.model.semantic_only:
            continue
        if  epoch%1 == 0 :
            print('eval...')
            with torch.no_grad():
                #ap_0 = validate(epoch, model, optimizer, val_loader, cfg, logger, writer,type=0)
                if 'isb' not in cfg.model.model_name:
                    ap_1 = validate(epoch, model, optimizer, val_loader, cfg, logger, writer,type=1)
                
                
                #ap_3 = validate(epoch, model, optimizer, val_loader, cfg, logger, writer,type=3)
                #if ap_0 > best_scores['pure_ray']:
                #    best_scores['pure_ray'] = ap_0
                if 'isb' not in cfg.model.model_name:
                    if ap_1 > best_scores['ray+exclusion_mask']:
                        best_scores['ray+exclusion_mask']=ap_1
                        best_epoch['ray+exclusion_mask']=epoch
                if 'isb' in cfg.model.model_name:
                    ap_0 = validate(epoch, model, optimizer, val_loader, cfg, logger, writer,type=0)
                    if ap_0 > best_scores['pure_ray']:
                        best_scores['pure_ray']=ap_0
                        best_epoch['pure_ray']=epoch

                '''
                ap_2 = validate(epoch, model, optimizer, val_loader, cfg, logger, writer,type=2)
                
                if ap_2 > best_scores['inclusion_mask']:
                    best_scores['inclusion_mask'] = ap_2
                    best_epoch['inclusion_mask'] = epoch
                '''
            print(best_epoch)
            
            cnt_eval += 1
        
        writer.flush()

    logger.info(f"Finish!!! Model at: {cfg.work_dir}")


if __name__ == "__main__":
    main()
