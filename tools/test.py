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
from spherical_mask.data import build_dataloader, build_dataset
from spherical_mask.evaluation import PointWiseEval, ScanNetEval
from spherical_mask.model import SphericalMask
from spherical_mask.model.criterion_spherical_mask import Criterion_SphericalMask 
from spherical_mask.util import (
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    get_dist_info,
    get_root_logger,
    init_dist,
    load_checkpoint_encoder,
    load_checkpoint_decoder
)

np.random.seed(0)
torch.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    return args



def test(model, val_loader, cfg, logger):
    logger.info("Test")
    all_pred_insts, all_sem_labels, all_ins_labels = [], [], []

    val_set = val_loader.dataset

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(val_set.CLASSES, dataset_name=cfg.data.train.type)

    torch.cuda.empty_cache()

   
    model.iterative_sampling = False
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)
            
            
            if i % 10 == 0:
                logger.info(f"Infer scene {i+1}/{len(val_set)}")
            
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
           

    

    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        miou, acc, mae = point_eval.get_eval(logger)

       

    else:
        logger.info("Evaluate instance segmentation")

        eval_res = scannet_eval.evaluate(all_pred_insts, all_sem_labels, all_ins_labels)
        del all_pred_insts, all_sem_labels, all_ins_labels

        

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

    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    logger.info(f"Mix precision training: {cfg.fp16}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    

    logger.info(f"Save at: {cfg.work_dir}")

    # model
    
    criterion = Criterion_SphericalMask(
        cfg.model.semantic_classes,
        cfg.model.instance_classes,
        cfg.model.semantic_weight,
        cfg.model.ignore_label,
        semantic_only=cfg.model.semantic_only,
        total_epoch=cfg.epochs,
        voxel_scale=cfg.data.train.voxel_cfg.scale,
    )
    
    model = SphericalMask(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=False).cuda()
    
   
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

    
    cfg.data.test['debug'] = False
    val_set = build_dataset(cfg.data.test, logger)
    
    test_loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)
   
    # optim
    default_lr = cfg.optimizer.lr  # default for batch 16
    _, world_size = get_dist_info()
    total_batch_size = cfg.dataloader.train.batch_size * world_size
    scaled_lr = default_lr * (total_batch_size / 16)
    cfg.optimizer.lr = scaled_lr
    logger.info(f"Scale LR from {default_lr} (batch size 16) to {scaled_lr} (batch size {total_batch_size})")
    
    # load_model
   
    
    assert args.ckpt != None, 'ckpt path must be provided for testing.'
    load_checkpoint_encoder(args.ckpt, logger, model)
    load_checkpoint_decoder(args.ckpt, logger, model)

    
    # test
    logger.info("Training")
    torch.cuda.empty_cache()
    ap = test(model, test_loader, cfg, logger)
   
    
        
if __name__ == "__main__":
    main()
