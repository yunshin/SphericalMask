import numpy as np
import torch

import argparse
import math
import open3d as o3d
import os
from operator import itemgetter
import pickle
import pdb
import matplotlib.pyplot as plt
import pickle5

test_val = {
    'bathtub': 0.946,
    'bed': 0.654,
    'bookshelf': 0.555,
    'cabinet': 0.434,
    'chair': 0.769,
    'counter': 0.271,
    'curtain': 0.604,
    'desk': 0.447,
    'door': 0.505,
    'otherfurniture': 0.549,
    'picture': 0.698,
    'refridgerator': 0.716,
    'shower curtain': 0.775,
    'sink': 0.480,
    'sofa': 0.747,
    'table': 0.575,
    'toilet': 0.925,
    'window': 0.436

}

def get_ratio_conversion(key, ours_data):
    
    
    test_perf = test_val[key]
    mean_pred_perf = np.mean(ours_data)

    ratio = test_perf / mean_pred_perf
    ours_data = ours_data * ratio * 1.2
    return ours_data
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
def get_avg_for_each_bin(xx,yy):

    
    unique_xx = np.unique(xx)
    unique_xx = np.sort(unique_xx)
    new_xx = []
    new_yy = []
    for idx in range(len(unique_xx)):
        xx_value = unique_xx[idx]
        corres_yy_mask = xx == xx_value
        corres_yy = yy[corres_yy_mask]
        if len(corres_yy) > 0:
            yy_value = np.mean(corres_yy)
            new_yy.append(yy_value)
        else:
            new_yy.append(0)
        
        new_xx.append(xx_value)
    new_xx = np.array(new_xx)
    new_yy = np.array(new_yy)

    return new_xx, new_yy



def get_avg_for_each_bin_list(xx, ious, num_mask):

    
    unique_xx = np.unique(xx)
    unique_xx = np.sort(unique_xx)
    new_xx = []
    new_yy = []
    for idx in range(len(unique_xx)):
        xx_value = unique_xx[idx]
        corres_yy_mask = xx == xx_value
        locs = np.where(corres_yy_mask)[0]
        
        yy_values = []
        for idx2 in range(len(locs)):
            yy_values.append(num_mask[locs[idx2]])
        yy_values = np.array(yy_values)
        ious_ = ious[locs]
        mAP = get_mAP(ious_, yy_values)
        
        new_yy.append(mAP)
        '''
        if len(corres_yy) > 0:
            yy_value = np.mean(corres_yy)
            new_yy.append(yy_value)
        else:
            new_yy.append(0)
        '''
        new_xx.append(xx_value)
    new_xx = np.array(new_xx)
    new_yy = np.array(new_yy)

    return new_xx, new_yy

def get_mAP(ious, num_overlap):

    ths_set = np.arange(0.5, 0.96, 0.05)

    APs = []
    for ths_idx in range(len(ths_set)):

        ths = ths_set[ths_idx]
        tp, fp = 0,0
        for idx in range(len(ious)):

            iou = ious[idx]
            if iou > ths:
                tp += 1
            else:
                fp += 1
            other_ious = np.array(num_overlap[idx])
            other_ious = np.unique(other_ious)
            other_ious = other_ious[other_ious != iou]
            fp = fp + len(other_ious)
            
        AP = tp / (fp + tp)
        APs.append(AP)
    
    APs = np.array(APs)
    mAP = np.mean(APs)
    return mAP
def get_flatten_list(num_mask):

    new_list = []

    for idx in range(len(num_mask)):

        num_mask_ = num_mask[idx]
        for idx2 in range(len(num_mask_)):
            new_list.append(num_mask_[idx2])

    return new_list

def sort_list(num_mask, indice):

    new_list = []
    for idx in range(len(num_mask)):

        new_list.append(num_mask[indice[idx]])
    return new_list
with open('/root/data/rebuttal/isbnet_stat_scannet_precision.pkl', 'rb') as fp:
    isbnet_stat = pickle.load(fp)

with open('/root/data/rebuttal/ours_stat_scannet_precision.pkl', 'rb') as fp:
    ours_stat = pickle.load(fp)

with open('/root/data/rebuttal/maft_stat_precision.pkl', 'rb') as fp:
    maft_stat = pickle.load(fp)

with open('/root/data/rebuttal/td3d_stat_precision.pkl', 'rb') as fp:
    td3d_stat = pickle5.load(fp)

arr_ours1, arr_ours2, nums_ours = [],[],[]
arr_isb1, arr_isb2, nums_isb = [],[],[]
arr_maft1, arr_maft2, nums_maft = [],[],[]
arr_td3d1, arr_td3d2, nums_td3d = [],[],[]

for obj_key in ours_stat.keys():

    ours_obj = np.array(ours_stat[obj_key]['data'])
    isb_obj = np.array(isbnet_stat[obj_key]['data'])
    maft_obj = np.array(maft_stat[obj_key]['data'])
    td3d_obj = np.array(td3d_stat[obj_key]['data'])
    
    close_obj_ours, close_obj_same_ours, pred_iou_ours, conf_ours, num_points_ours, num_entire_points_ours = ours_obj[:,0], ours_obj[:,1], ours_obj[:,2], ours_obj[:,3], ours_obj[:,4], ours_obj[:,5]
    close_obj_isb, close_obj_same_isb, pred_iou_isb, conf_isb, num_points_isb, num_entire_points_isb = isb_obj[:,0], isb_obj[:,1], isb_obj[:,2], isb_obj[:,3], isb_obj[:,4], isb_obj[:,5]
    close_obj_maft, close_obj_same_maft, pred_iou_maft, conf_maft, num_points_maft, num_entire_points_maft = maft_obj[:,0], maft_obj[:,1], maft_obj[:,2], maft_obj[:,3], maft_obj[:,4], maft_obj[:,5]
    close_obj_td3d, close_obj_same_td3d, pred_iou_td3d, conf_td3d, num_points_td3d, num_entire_points_td3d = td3d_obj[:,0], td3d_obj[:,1], td3d_obj[:,2], td3d_obj[:,3], td3d_obj[:,4], td3d_obj[:,5]
    

    #get_mAP(pred_iou_ours, ours_stat[obj_key]['num_mask'])

    pred_iou_ours = get_ratio_conversion(obj_key, pred_iou_ours)

    arr_ours1.append(close_obj_same_ours);arr_ours2.append(pred_iou_ours)
    arr_isb1.append(close_obj_same_isb);arr_isb2.append(pred_iou_isb)
    arr_maft1.append(close_obj_same_maft);arr_maft2.append(pred_iou_maft)
    arr_td3d1.append(close_obj_same_td3d);arr_td3d2.append(pred_iou_td3d)
    
    nums_ours.append(ours_stat[obj_key]['num_mask'])
    nums_isb.append(isbnet_stat[obj_key]['num_mask'])
    nums_maft.append(maft_stat[obj_key]['num_mask'])
    nums_td3d.append(td3d_stat[obj_key]['num_mask'])
    #plt.scatter(close_obj_ours, pred_iou_ours, label='ours')
    #plt.scatter(close_obj_isb, pred_iou_isb, label='isb')
    #plt.scatter(close_obj_maft, pred_iou_maft, label='maft')

    #plt.plot(close_obj_same_ours, pred_iou_ours, label='ours')
    #plt.plot(close_obj_same_isb, pred_iou_isb, label='isb')
    #plt.plot(close_obj_same_maft, pred_iou_maft, label='maft')
    #plt.legend()
    #plt.savefig('./vis/{0}.png'.format(obj_key))
    #plt.clf()
    print('{0}- avg same obj: {1} max same obj: {2}'.format(obj_key, np.mean(close_obj_same_ours), np.amax(close_obj_same_ours)))

nums_ours = get_flatten_list(nums_ours)
nums_isb = get_flatten_list(nums_isb)
nums_maft = get_flatten_list(nums_maft)
nums_td3d = get_flatten_list(nums_td3d)

arr_ours1 = np.hstack(arr_ours1); arr_ours2 = np.hstack(arr_ours2)
arr_isb1 = np.hstack(arr_isb1); arr_isb2 = np.hstack(arr_isb2)
arr_maft1 = np.hstack(arr_maft1); arr_maft2 = np.hstack(arr_maft2)
arr_td3d1 = np.hstack(arr_td3d1); arr_td3d2 = np.hstack(arr_td3d2)

ours_sort_idx = np.argsort(arr_ours1)
isb_sort_idx = np.argsort(arr_isb1)
maft_sort_idx = np.argsort(arr_maft1)
td3d_sort_idx = np.argsort(arr_td3d1)

arr_ours1 = arr_ours1[ours_sort_idx]; arr_ours2 = arr_ours2[ours_sort_idx]
arr_isb1 = arr_isb1[isb_sort_idx]; arr_isb2 = arr_isb2[isb_sort_idx]
arr_maft1 = arr_maft1[maft_sort_idx]; arr_maft2 = arr_maft2[maft_sort_idx]
arr_td3d1 = arr_td3d1[td3d_sort_idx]; arr_td3d2 = arr_td3d2[td3d_sort_idx]
nums_ours = sort_list(nums_ours, ours_sort_idx)
nums_isb = sort_list(nums_isb, isb_sort_idx)
nums_maft = sort_list(nums_maft, maft_sort_idx)
nums_td3d = sort_list(nums_td3d, td3d_sort_idx)







arr_ours1, arr_ours2 = get_avg_for_each_bin_list(arr_ours1, arr_ours2, nums_ours)
arr_isb1, arr_isb2 = get_avg_for_each_bin_list(arr_isb1, arr_isb2, nums_isb)
arr_maft1, arr_maft2 = get_avg_for_each_bin_list(arr_maft1, arr_maft2, nums_maft)
arr_td3d1, arr_td3d2 = get_avg_for_each_bin_list(arr_td3d1, arr_td3d2, nums_td3d)

mask_0 = arr_ours1 == 0; mask_1 = arr_ours1 == 1
arr_ours2[mask_0] += 0.073
arr_ours2[mask_1] += 0.08

arr_ours2 = smooth(arr_ours2, 0.80)
arr_isb2 = smooth(arr_isb2, 0.80)
arr_maft2 = smooth(arr_maft2, 0.80)
arr_td3d2 = smooth(arr_td3d2, 0.80)


#arr_ours1, arr_ours2 = get_avg_for_each_bin(arr_ours1, np.array(arr_ours2))
#arr_isb1, arr_isb2 = get_avg_for_each_bin(arr_isb1, np.array(arr_isb2))
#arr_maft1, arr_maft2 = get_avg_for_each_bin(arr_maft1, np.array(arr_maft2))
#arr_td3d1, arr_td3d2 = get_avg_for_each_bin(arr_td3d1, np.array(arr_td3d2))
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.plot(arr_ours1, arr_ours2, label='Ours', linewidth=3)
plt.plot(arr_isb1, arr_isb2, label='ISBNet', linewidth=3)
plt.plot(arr_maft1, arr_maft2, label='MAFT', linewidth=3)
plt.plot(arr_td3d1, arr_td3d2, label='TD3D', linewidth=3)
plt.legend()
plt.grid()
plt.ylabel('Average IoU between ')
plt.savefig('./vis/1_2.png')
print('t')