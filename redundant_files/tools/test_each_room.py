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
from isbnet.util import get_root_logger, init_dist, load_checkpoint, rle_decode, rle_encode_gpu
from isbnet.model import PolarNet_Dyco_Ray_Cluster
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma
from isbnet.model import PolarNet_Dyco_Ray_Cluster_Gamma_Ablation
from isbnet.model.criterion_spherical_mask import Criterion_Dyco_Ray_Cluster_Gamma_Ablation
from joblib import Parallel, delayed

import pdb
import pickle
def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--load_path", type=str, help="directory for output results")
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
        np.savetxt(mask_path, mask, fmt="%d")
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
def load_pred_new(base_path, scan_id, benchmark_indice, gt=None):
    instances = []
    
    
    #if os.path.exists(os.path.join(base_path,scan_id+'.txt')) :
    if os.path.exists(os.path.join(base_path,scan_id+'_new.txt')) :
        print('in')
        is_load = True
        #fp = open(os.path.join(base_path,scan_id+'.txt'),'r')
        fp = open(os.path.join(base_path,scan_id+'_new.txt'),'r')
        lines = fp.readlines()
        fp.close()
        #pdb.set_trace()
        benchmark_indice_array = np.array(benchmark_indice)
        for idx in range(len(lines)):

            line_info = lines[idx].split(' ')
            pred_mask_path, label_num_benchmark, conf = line_info[0], int(line_info[1]), float(line_info[2])
            
           
            mask_path = os.path.join(base_path,pred_mask_path)
            
           
            mask = np.loadtxt(mask_path).astype(np.int32)

            pred = {}
            pred['scan_id'] = scan_id
            

            
            pred_label = np.where(benchmark_indice_array == label_num_benchmark)[0][0]
            pred_label = pred_label -1

            pred['label_id'] = pred_label#label_num_benchmark
            pred['conf'] = conf
            pred['pred_mask'] = rle_encode_gpu(torch.tensor(mask).float().cuda())
            instances.append(pred)
    else:
        is_load = False
    return instances, is_load
def par_func(filename):
    
    #t_str="%.4d" % counter   
    #filename = 'file_'+t_str+'.dat'
   
    temp_array = np.loadtxt(filename).astype(np.int32)
    #temp_array.shape=[N1,N2]
    # temp_array = np.random.randn(N1, N2)  # use this line to test
    return temp_array

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


def pred_to_benchmark_sem(pred):
    
    benchmark_cls_num = PRED2SEM[pred]
    return benchmark_cls_num

def load_pred(base_path, scan_id, benchmark_indice):
    instances = []
    
    
    #if os.path.exists(os.path.join(base_path,scan_id+'.txt')) :
    all_masks, all_cls, all_scores = [],[], []
    if os.path.exists(os.path.join(base_path,scan_id+'.txt')) :
        print('in')

        
        is_load = True
        #fp = open(os.path.join(base_path,scan_id+'.txt'),'r')
        fp = open(os.path.join(base_path,scan_id+'.txt'),'r')
        lines = fp.readlines()
        fp.close()

        benchmark_indice_array = np.array(benchmark_indice)

        
        num_jobs = 20
        mask_paths = []
        for idx in range(len(lines)):
            
            line_info = lines[idx].split(' ')
            pred_mask_path, label_num_benchmark, conf = line_info[0], int(line_info[1]), float(line_info[2])
            mask_path = os.path.join(base_path,pred_mask_path)
            mask_paths.append(mask_path)
       
        masks_list = Parallel(n_jobs=num_jobs)(delayed(par_func) 
                                                ( counter)
                                                for counter in mask_paths) 
       
        for idx in range(len(lines)):

            line_info = lines[idx].split(' ')
            pred_mask_path, label_num_benchmark, conf = line_info[0], int(line_info[1]), float(line_info[2])
            
            
            mask_path = os.path.join(base_path,pred_mask_path)
            mask = masks_list[idx]
            #mask = np.loadtxt(mask_path).astype(np.int32)

            if idx == 0:
                all_masks = torch.zeros(len(lines),mask.shape[0])
                all_cls = torch.zeros(len(lines))
                all_scores = torch.zeros(len(lines))
            all_masks[idx] = torch.tensor(mask).float()
            
            all_scores[idx] = conf
            pred = {}
            pred['scan_id'] = scan_id
            

            #label_num_benchmark = pred_to_benchmark_sem(label_num_benchmark)
            #pred_label = label_num_benchmark
            try:
                pred_label = np.where(benchmark_indice_array == label_num_benchmark)[0][0]
            except:
                pdb.set_trace()
            pred_label = pred_label -1


            all_cls[idx] = pred_label

            pred['label_id'] = pred_label#label_num_benchmark
            pred['conf'] = conf
            pred['pred_mask'] = rle_encode_gpu(torch.tensor(mask).float().cuda())
            instances.append(pred)
       
    else:
        is_load = False
    return instances, is_load, all_masks, all_cls, all_scores
def load_pred_old(base_path, scan_id, benchmark_indice):
    instances = []
    
    
    #if os.path.exists(os.path.join(base_path,scan_id+'.txt')) :
    if os.path.exists(os.path.join(base_path,scan_id+'.txt')) :
        print('in')
        is_load = True
        #fp = open(os.path.join(base_path,scan_id+'.txt'),'r')
        fp = open(os.path.join(base_path,scan_id+'.txt'),'r')
        lines = fp.readlines()
        fp.close()

        benchmark_indice_array = np.array(benchmark_indice)
        for idx in range(len(lines)):

            line_info = lines[idx].split(' ')
            pred_mask_path, label_num_benchmark, conf = line_info[0], int(line_info[1]), float(line_info[2])
            
           
            mask_path = os.path.join(base_path,pred_mask_path)
            
           
            mask = np.loadtxt(mask_path).astype(np.int32)

            pred = {}
            pred['scan_id'] = scan_id
            

            pdb.set_trace()
            pred_label = np.where(benchmark_indice_array == label_num_benchmark)[0][0]
            pred_label = pred_label -1

            pred['label_id'] = pred_label#label_num_benchmark
            pred['conf'] = conf
            pred['pred_mask'] = rle_encode_gpu(torch.tensor(mask).float().cuda())
            instances.append(pred)
    else:
        is_load = False
    return instances, is_load
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
        type_num = 0
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

   

    time_arr = []

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(dataset.CLASSES, dataset_name=cfg.data.train.type)


    load_path = args.load_path
    '''
    if load_path is not None:
        new_scans = []
        new_list = os.listdir(load_path+'/new_predicted_masks')

        for idx in range(len(new_list)):
            file_name = new_list[idx]
            
            new_scan_id = file_name[:12]#.split('_')[0]
            if new_scan_id not in new_scans:
                new_scans.append(new_scan_id)
    
    '''
    if cfg.data.test.type == "s3dis":
        s3dis_eval = S3DISEval()
    data_length = len(dataloader)

    for idx_room in range(data_length):

        scan_ids, sem_preds, offset_preds, offset_vertices_preds = [], [], [], []
        nmc_clusters = []
        pred_insts, sem_labels, ins_labels = [], [], []
        object_conditions = []
        gt_insts = []
        
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dataloader):

                scan_id = batch['scan_ids'][0]
        
                #if scan_id not in new_scans:
                #    continue

                #if i != 5:
                #    continue

                #if i > 5:
                #    break
                #if scan_id != 'scene0019_01':
                #    continue
                print('{0}: {1}'.format(i, scan_id))
                #pdb.set_trace()
                t1 = time.time()
                batch['type'] = type_num
                # NOTE avoid OOM during eval s3dis with full resolution
                if cfg.data.test.type == "s3dis":
                    torch.cuda.empty_cache()

                #if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 1400000:
                if cfg.data.train.type == "s3dis" and batch["coords_float"].shape[0] > 1200000:    
                    continue
                
                

                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    res = model(batch)
                
                
                
                #res = {}
                #res['scan_id'] = batch['scan_ids']
                '''
                ############Loading part
                is_load = False
                #pred_insts_load, is_load = load_pred_new(load_path, scan_id, dataset.BENCHMARK_SEMANTIC_IDXS)
                pred_insts_load, is_load, _,_,_ = load_pred(load_path, scan_id, dataset.BENCHMARK_SEMANTIC_IDXS)
                if is_load == True:
                    print('')
                    #res["pred_instances"] = pred_insts_load
                else:
                    continue
                ############
                '''
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
                    #pred_insts, sem_labels, ins_labels = [], [], []
                    gt_inst = get_gt_instances(torch.tensor(res['semantic_labels']), torch.tensor(res['instance_labels']))
                    pred_insts.append(res["pred_instances"])
                    sem_labels.append(res["semantic_labels"])
                    ins_labels.append(res["instance_labels"])
                    gt_insts.append(gt_inst)

                    '''
                    ############################
                    perf = scannet_eval.evaluate(pred_insts, sem_labels, ins_labels)        
                    
                    room_name = batch['scan_ids'][0]
                    out_path = args.out + '/performance/' + room_name + '_perf.pkl'
                    with open(out_path, 'wb') as handle:
                        pickle.dump(perf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print('{0}: {1}'.format(i,out_path))
                    #############################
                    '''
                    #scannet_eval.evaluate(pred_insts, sem_labels, ins_labels) # For finding which scene to visualize
                if cfg.save_cfg.object_conditions:
                    object_conditions.append(res["object_conditions"])
                if cfg.save_cfg.offset_vertices:
                    offset_vertices_preds.append(res["offset_vertices_preds"])
                if cfg.save_cfg.semantic:
                    sem_preds.append(res["semantic_preds"])
                if cfg.save_cfg.offset:
                    offset_preds.append(res["offset_preds"])
                #break
        perf = scannet_eval.evaluate(pred_insts, sem_labels, ins_labels)        
        pdb.set_trace()
        room_name = batch['scan_ids'][0]
        
        #logger.info(f"Average run time: {mean_time:.4f}")

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
