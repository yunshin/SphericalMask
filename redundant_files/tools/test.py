import numpy as np
import torch
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel

import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial
from isbnet.data import build_dataloader, build_dataset
from isbnet.evaluation import PointWiseEval, S3DISEval, ScanNetEval
from isbnet.model import ISBNet
from isbnet.util import get_root_logger, init_dist, load_checkpoint, rle_decode
from isbnet.model import PolarNet_Dyco_Ray_Cluster
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma_Ablation
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma_Ablation_Binary
from isbnet.model.criterion_spherical_mask import Criterion_Dyco_Ray_Cluster_Gamma_Ablation
import pdb
def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--out", type=str, help="directory for output results")
    parser.add_argument("--save_lite", action="store_true")
    parser.add_argument("--only_backbone", action="store_true", help="only train backbone")
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.npy") for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts, benchmark_sem_id):
    f = open(osp.join(root, f"{scan_id}.txt"), "w")
    os.makedirs(osp.join(root, "predicted_masks"), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst["scan_id"]

        # NOTE process to map label id to benchmark
        label_id = inst["label_id"]  # 1-> 18
        label_id = label_id + 1  # 2-> 19 , 0,1: background
        label_id = benchmark_sem_id[label_id]

        conf = inst["conf"]

        f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n")
        # f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f} " + box_string + "\n")
        mask_path = osp.join(root, "predicted_masks", f"{scan_id}_{i:03d}.txt")
        mask = rle_decode(inst["pred_mask"])
        wrtie_mask_txt(mask_path, mask)
        #np.savetxt(mask_path, mask, fmt="%d")
    f.close()



def save_pred_instances(root, name, scan_ids, pred_insts, benchmark_sem_id):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    benchmark_sem_ids = [benchmark_sem_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, benchmark_sem_ids))
    pool.close()
    pool.join()

def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')

def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()

def get_gt_instances(semantic_labels, instance_labels):
    """Get gt instances for evaluation."""
    # convert to evaluation format 0: ignore, 1->N: valid
    semantic_classes=20
    instance_classes=18

    
    label_shift = semantic_classes - instance_classes
    semantic_labels = semantic_labels - label_shift + 1
    semantic_labels[semantic_labels < 0] = 0
    instance_labels += 1
    ignore_inds = instance_labels < 0
    # scannet encoding rule
    gt_ins = semantic_labels * 1000 + instance_labels
    gt_ins[ignore_inds] = 0
    gt_ins = gt_ins.cpu().numpy()
    return gt_ins
def wrtie_mask_txt(file_path, mask):
    fp_mask = open(file_path,'w')

    for idx in range(len(mask)):
        fp_mask.write('{0}\n'.format(str(int(mask[idx]))))
    fp_mask.close()

def chk_ratio(points, instance_labels, semantic_labels):

    inst_num = instance_labels.max() + 1
    size_of_points = len(points)
    ratios = []
    
    for inst_idx in range(inst_num):

        inst_mask = instance_labels == inst_idx
        sem_label = semantic_labels[inst_mask][0]
        if sem_label > 0:
            num_foregrounds = inst_mask.sum()
            ratios.append(num_foregrounds.item() / size_of_points)

    return ratios


PRED2SEM = {
    1:3,
    2:4,
    3:5,
    4:6,
    5:7,
    6:8,
    7:9,
    8:10,
    9:11,
    10:12,
    11:14,
    12:16,
    13:24,
    14:28,
    15:33,
    16:34,
    17:36,
    18:39
}
label_to_name={#from prediction
    3:'cabinet',
    4:'bed',
    5:'chair',
    6:'sofa',
    7:"table",
    8:"door",
    9:"window",
    10:"bookshelf",
    11:"picture",
    12:"counter",
    14:"desk",
    16:"curtain",
    24:"refridgerator",
    28:"shower curtain",
    33:"toilet",
    34:"sink",
    36:"bathtub",
    39:"otherfurniture",
}
label_to_name_s3dis = {
        0:"ceiling",
        1:"floor",
        2:"wall",
        3:"beam",
        4:"column",
        5:"window",
        6:"door",
        7:"chair",
        8:"table",
        9:"bookcase",
        10:"sofa",
        11:"board",
        12:"clutter"
}
def get_density(points, inst_label, sem_label, benchmark_id, dist_ths=0.1, ratio_ths=0.1, s3dis=False, only_cls='none'):

    instance_num = inst_label.max().item() + 1
    inst_masks = {}

    for idx in range(instance_num):

        inst_mask = inst_label == idx
        if inst_mask.sum().item() == 0:
            continue 
        sem_label_ = sem_label[inst_mask][0].item()
        if s3dis == False:
            if sem_label_ == 0:
                continue
        else:

            if inst_label[inst_mask][0].item() < 0 or sem_label_ < 0:
                continue
        if only_cls != 'none':
            inst_name = label_to_name_s3dis[sem_label_]
            if inst_name != only_cls:
                continue
        inst_points = points[inst_mask]
        inst_masks[idx] = {}
        inst_masks[idx]['points'] = inst_points
        inst_masks[idx]['close_inst'] = 0
        inst_masks[idx]['close_inst_same'] = 0
        inst_masks[idx]['iou'] = 0
        inst_masks[idx]['conf'] = 0
        inst_masks[idx]['num_mask'] = []
        inst_masks[idx]['pred_mask'] = []
        
        #PRED2SEM[sem_label_]
        if s3dis == False:
            if sem_label_ not in PRED2SEM:
                pdb.set_trace()
            inst_masks[idx]['label'] = PRED2SEM[sem_label_]
            inst_name = label_to_name[PRED2SEM[sem_label_]]
        else:
            inst_masks[idx]['label'] = sem_label_ #- 1
            inst_name = label_to_name_s3dis[sem_label_]
        inst_masks[idx]['inst_type'] = inst_name
    num_keys = list(inst_masks.keys())
    cnt = 0
    for inst_key in inst_masks.keys():
        
        query_points = inst_masks[inst_key]['points']
        center = torch.mean(query_points,0)
        start_time = time.time()
        if len(query_points) > 200000:
            print('query_points : {0}'.format(len(query_points)))
            continue
        #dist_from_center = torch.cdist(query_points, center[None,:], p=2)[:,0]
        for inst_key2 in inst_masks.keys():
            if inst_key == inst_key2:
                continue
            
            points = inst_masks[inst_key2]['points']
            if len(points) > 200000:
                continue
            #print('query_points : {0} points : {1}'.format(len(query_points), len(points)))
            num_points_to = 10000
            num_divider = int(len(points) / num_points_to) + 1
            dist = []
            for idx_divide in range(num_divider):
                if idx_divide == (num_divider - 1):
                    points_ = points[idx_divide*num_points_to : ]
                else:
                    points_ = points[idx_divide*num_points_to : (idx_divide+1)*num_points_to]
                dist_ = torch.cdist(query_points, points_, p=2)
                dist.append(dist_)
            
            dist = torch.cat(dist,1)
            
            #dist = torch.cdist(query_points, points, p=2)
            
            dist = torch.amin(dist,1)
            
            #pdb.set_trace()
            #num_close_other = dist_from_center < dist
            num_close_other = (dist < dist_ths) * (dist > 0)
            ratio  = num_close_other.sum().item() / len(query_points)

            #if ratio > ratio_ths:
            if num_close_other.sum().item() > 0:
                inst_masks[inst_key]['close_inst'] += 1
                if inst_masks[inst_key]['label'] == inst_masks[inst_key2]['label']: 
                 
                    inst_masks[inst_key]['close_inst_same'] += 1
        print('{0}/{1} time : {2}'.format(cnt, len(num_keys),time.time() - start_time))
        cnt+=1
    
    return inst_masks
def get_precision(pred_mask, pred_label, conf, inst_label, sem_label, benchmark_id, iou_dict, s3dis=False):

    instance_num = inst_label.max().item() + 1
    if s3dis == False:
        pred_label += 1
        pred_label = benchmark_id[pred_label]
    
    for idx in range(instance_num):

        inst_mask = inst_label == idx
        if inst_mask.sum().item() == 0:
            continue 
        if idx not in iou_dict:
            continue
        #sem_label_ = sem_label[inst_mask][0].item()
        inst_mask = inst_mask*1.0
        sem_label_ = iou_dict[idx]['label']
        if sem_label_ == 0:
            continue
        
        if idx not in iou_dict:
            iou_dict[idx] = {}
            iou_dict[idx]['iou'] = 0
            iou_dict[idx]['conf'] = 0
            #iou_dict[idx]['label'] = sem_label_
            iou_dict[idx]['pred_mask'] = []
            iou_dict[idx]['num_mask'] = []
        if sem_label_ == pred_label:

            intersection = (inst_mask) * pred_mask
            intersection = intersection.sum()
            iou = intersection / (inst_mask.sum() + pred_mask.sum() - intersection + 1e-6)
            
            if iou.item() > iou_dict[idx]['iou']:
                
                iou_dict[idx]['iou'] = iou.item()
                iou_dict[idx]['conf'] = conf
                if len(iou_dict[idx]['pred_mask']) == 0:
                    iou_dict[idx]['pred_mask'] = pred_mask
            if iou.item() > 0:
                iou_dict[idx]['num_mask'].append(iou.item())
    return iou_dict



def get_cls_acc(pred, batch, gt_insts, benchmark_id, s3dis=False, only_cls='none'):
    
    #inst_label = batch['instance_labels']
    #sem_label = batch['semantic_labels']
    if s3dis == False:

        sem_label = gt_insts // 1000    
        inst_label = gt_insts % 1000
    else:
        sem_label = gt_insts[0]
        inst_label = gt_insts[1]
    
    pred_instances = pred['pred_instances']
    #iou_dict = {}
    
    iou_dict = get_density(batch['coords_float'], inst_label, sem_label, benchmark_id, s3dis=s3dis, only_cls=only_cls)
    
    for pred_idx in range(len(pred_instances)):
        
        pred_mask = pred_instances[pred_idx]['pred_mask']
        pred_label = pred_instances[pred_idx]['label_id']
        conf = pred_instances[pred_idx]['conf']
        pred_mask = rle_decode(pred_mask)
        
        if s3dis:
            pred_label = pred_label -1
        iou_dict = get_precision(pred_mask, pred_label, conf, inst_label, sem_label, benchmark_id, iou_dict, s3dis)
        
    return iou_dict
def chk_all_cls_s3dis(stat_dict):

    is_perfect = True
    print(stat_dict)
    for key_idx in label_to_name_s3dis:
        name = label_to_name_s3dis[key_idx]
        
        if name not in stat_dict:
            print('{0} not in dict'.format(name))
            is_perfect = False
            break
    return is_perfect
def merge_stat(iou_dict, stat_dict):

    for inst_key in iou_dict:

        pred = iou_dict[inst_key]
        close_inst, pred_iou, conf, inst_name = pred['close_inst'], pred['iou'], pred['conf'], pred['inst_type']
        close_inst_same = pred['close_inst_same']
        num_obj_points = len(pred['points'])
        num_entire_points = len(pred['pred_mask'])
        
        if inst_name not in stat_dict:
            stat_dict[inst_name] = {}
            stat_dict[inst_name]['data'] = []
            stat_dict[inst_name]['num_mask'] = []
        
        stat_dict[inst_name]['num_mask'].append(pred['num_mask'])
        stat_dict[inst_name]['data'].append([close_inst, close_inst_same, pred_iou, conf, num_obj_points, num_entire_points])
    return stat_dict

    
def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    if args.only_backbone:
        logger.info("Only test backbone")
        cfg.model.semantic_only = True
    if cfg.model.model_name == 'isbnet':
        model = ISBNet(**cfg.model, dataset_name=cfg.data.train.type).cuda()
        type_num = 2
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster':
        
        model = PolarNet_Dyco_Ray_Cluster(**cfg.model, dataset_name=cfg.data.train.type).cuda()
        type_num = 2
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma':
        
        model = PolarNet_Dyco_Ray_Cluster_Gamma(**cfg.model, dataset_name=cfg.data.train.type).cuda()
        type_num = 1

    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation':
        
        model = PolarNet_Dyco_Ray_Cluster_Gamma_Ablation(**cfg.model, dataset_name=cfg.data.train.type).cuda()
        type_num = 1
        #type_num = 2 #for ray refine
        #type_num = 0 # for ray ablation
    elif cfg.model.model_name == 'polarnet_dyco_ray_cluster_gamma_ablation_binary':
        
        model = PolarNet_Dyco_Ray_Cluster_Gamma_Ablation_Binary(**cfg.model, dataset_name=cfg.data.train.type).cuda()
        type_num = 1
        #type_num = 2 #for ray refine
        #type_num = 0 # for ray ablation
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

    scan_ids, sem_preds, offset_preds, offset_vertices_preds = [], [], [], []
    nmc_clusters = []
    pred_insts, sem_labels, ins_labels = [], [], []
    object_conditions = []
    gt_insts = []
    time_arr = []
    stat_dict = {}
    chk_dict = {}
    mem_list = []
    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(dataset.CLASSES, dataset_name=cfg.data.train.type)

    if cfg.data.test.type == "s3dis":
        s3dis_eval = S3DISEval()

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            t1 = time.time()
            batch['type'] = type_num
            '''
            ratios = chk_ratio(batch['coords_float'], batch['instance_labels'], batch['semantic_labels'])
            fp = open('foreground_ratios.txt','a+')
            for idx_f in range(len(ratios)):
                fp.write('{0}\n'.format(ratios[idx_f]))
            fp.close()
            print(i)
            continue
            '''
            # NOTE avoid OOM during eval s3dis with full resolution
            if cfg.data.test.type == "s3dis":
                torch.cuda.empty_cache()

            #if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 1400000:
            if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 1400000:    
                continue
            #print(batch['scan_ids'][0])
            #if batch['scan_ids'][0] != 'Area_5_office_14':
            #    continue
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)
            '''
            free_mem, total_mem = torch.cuda.mem_get_info()
            mem_usage = (total_mem - free_mem)/1000000
            print('memory usage: {0}'.format(mem_usage))
            mem_list.append(mem_usage)
            if i > 10:
                mem_usage = np.array(mem_list)
                print(np.mean(mem_usage))
                pdb.set_trace()
            '''
            scan_id = batch['scan_ids'][0]
            is_save = False
            if is_save:
                save_path = '/root/data/pcdet_data/stpls3d_val_pcdrgb/'
                points_np = res['coords_float']
                rgb_np = res['feats_np']
                prgb = np.concatenate([points_np, rgb_np],1)
                
                np.save('{0}_{1}_prgb.npy'.format(save_path, scan_id),prgb)
            
            #mask = rle_decode(res['pred_instances'][0]['pred_mask'])
            #wrtie_mask_txt('/root/data/tmp.txt', mask)
            #
            

            t2 = time.time()
            time_arr.append(t2 - t1)

            if i % 10 == 0:
                logger.info(f"Infer scene {i+1}/{len(dataset)}")
            # for res in result:
            scan_ids.append(res["scan_id"])
            
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
               
                gt_inst = get_gt_instances(torch.tensor(res['semantic_labels']), torch.tensor(res['instance_labels']))
                pred_insts.append(res["pred_instances"])
                sem_labels.append(res["semantic_labels"])
                ins_labels.append(res["instance_labels"])
                gt_insts.append(gt_inst)
                
                
                s3dis = False
                '''
                print(i)
                if s3dis:
                    #if i < 5:
                    #    continue
                    gt_inst = [res["semantic_labels"], res["instance_labels"]]
                    only_cls = 'chair'
                    iou_dict = get_cls_acc(res, batch, gt_inst, dataset.BENCHMARK_SEMANTIC_IDXS, s3dis=True, only_cls=only_cls)
                    
                else:
                    only_cls = 'none'
                    iou_dict = get_cls_acc(res, batch, gt_inst, dataset.BENCHMARK_SEMANTIC_IDXS, s3dis=False, only_cls=only_cls)
                #if s3dis:
                #    stat_dict = {}
                stat_dict = merge_stat(iou_dict, stat_dict)
                
                if s3dis:
                    for key_stat in stat_dict:
                        if key_stat not in chk_dict:
                            chk_dict[key_stat] = 1
                        else:
                            chk_dict[key_stat] += 1
                    if only_cls != 'none' and only_cls in chk_dict:
                        print('{0}: {1}'.format(only_cls, chk_dict[only_cls]))
                        if chk_dict[only_cls] > 10:

                            import pickle
                            save_dict_path = '/root/data/rebuttal/ours_stat_{0}_s3dis.pkl'.format(only_cls)
                            with open(save_dict_path, 'wb') as handle:
                                pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            break
                    #pdb.set_trace()    
                
                '''
                if s3dis and chk_all_cls_s3dis(chk_dict):
                    break
                
                #scannet_eval.evaluate(pred_insts, sem_labels, ins_labels) # For finding which scene to visualize
            if cfg.save_cfg.object_conditions:
                object_conditions.append(res["object_conditions"])
            if cfg.save_cfg.offset_vertices:
                offset_vertices_preds.append(res["offset_vertices_preds"])
            if cfg.save_cfg.semantic:
                sem_preds.append(res["semantic_preds"])
            if cfg.save_cfg.offset:
                offset_preds.append(res["offset_preds"])
    '''
    import pickle
    save_dict_path = '/root/data/rebuttal/isbnet_stat_scannet_precision.pkl'
    with open(save_dict_path, 'wb') as handle:
        pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pdb.set_trace()    
    '''
    '''
    import pickle
    save_dict_path = '/root/data/rebuttal/ours_stat_{0}_s3dis.pkl'.format(only_cls)
    with open(save_dict_path, 'wb') as handle:
        pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_dict_path)
    print('done')
    pdb.set_trace()
    '''
    # NOTE eval final inst mask+box
    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        point_eval.get_eval(logger)

    else:
        logger.info("Evaluate instance segmentation")
        scannet_eval.evaluate(pred_insts, sem_labels, ins_labels)

        if cfg.data.test.type == "s3dis":
            logger.info("Evaluate instance segmentation by S3DIS metrics")
            s3dis_eval.evaluate(pred_insts, sem_labels, ins_labels)

    mean_time = np.array(time_arr).mean()
    logger.info(f"Average run time: {mean_time:.4f}")

    # save output
    if not args.out:
        return

    logger.info("Save results")
    if cfg.save_cfg.semantic:
        save_npy(args.out, "semantic_pred", scan_ids, sem_preds)
    if cfg.save_cfg.offset:
        save_npy(args.out, "offset_pred", scan_ids, offset_preds)
    if cfg.save_cfg.offset_vertices:
        save_npy(args.out, "offset_vertices_pred", scan_ids, offset_vertices_preds)
    if cfg.save_cfg.object_conditions:
        save_npy(args.out, "object_conditions", scan_ids, object_conditions)
    if cfg.save_cfg.instance:
        save_pred_instances(args.out, "pred_instance", scan_ids, pred_insts, dataset.BENCHMARK_SEMANTIC_IDXS)
        save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts)
    if cfg.save_cfg.nmc_clusters:
        save_npy(args.out, "nmc_clusters_ballquery", scan_ids, nmc_clusters)


if __name__ == "__main__":
    main()
