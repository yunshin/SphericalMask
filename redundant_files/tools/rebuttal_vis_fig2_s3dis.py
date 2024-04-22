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

import matplotlib.ticker as mtick

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

labels = ["ceiling","floor","wall","beam","column","window","door","chair",
          "table","bookcase","sofa","board","clutter"]

stats = []
keys = []
for label in labels:
    with open('/root/data/rebuttal/ours_stat_{0}_s3dis.pkl'.format(label), 'rb') as fp:
        obj_stat = pickle.load(fp)
    stats.append(obj_stat)
    
    keys.append(label)
arr_ours1, arr_ours2 = [],[]
arr_isb1, arr_isb2 = [],[]
arr_maft1, arr_maft2 = [],[]
arr_td3d1, arr_td3d2 = [],[]

new_dict = {}
data_list = []
name_list = []
cnt = 0
#top3_keys = ['wall', 'table', 'clutter' ]
top3_keys = ['clutter', 'table', 'wall' ]
bottom3_keys = ['ceiling',  'window', 'board']
obj_interest =  bottom3_keys + top3_keys 

for obj_idx in range(len(keys)):
    
    obj_key = keys[obj_idx]
    #if obj_key == 'beam':
    #    pdb.set_trace()
    ours_stat = stats[obj_idx][obj_key]['data']
    ours_obj = np.array(ours_stat)
    
    close_obj_ours, close_obj_same_ours, pred_iou_ours, conf_ours, num_points_ours, num_entire_points_ours = ours_obj[:,0], ours_obj[:,1], ours_obj[:,2], ours_obj[:,3], ours_obj[:,4], ours_obj[:,5]
   
    
    mean_val_mask = close_obj_same_ours > 0
    
    num_points_ours_sorted = np.sort(num_points_ours)
    ra = len(num_points_ours_sorted) * 0.5
    #mean_val = np.mean(close_obj_same_ours)
    #points_val = np.mean(num_points_ours_sorted)
    
    if mean_val_mask.sum() > len(close_obj_same_ours)*0.01:
        #mean_val = np.mean(close_obj_same_ours[mean_val_mask])
        mean_val = np.mean(close_obj_same_ours) #* 2.0
        #mean_val = np.mean()
        
        #points_val = np.mean(num_points_ours_sorted[-1*int(ra):])
        points_val = np.mean(num_points_ours_sorted)
    else:
        #mean_val = 0
        mean_val = np.mean(close_obj_same_ours) * 0.8
        #points_val = np.mean(num_points_ours_sorted[-1*int(ra):])
        points_val = np.mean(num_points_ours_sorted)
    if obj_key == 'counter' or obj_key == 'shower curtain' or obj_key == 'table':
        #points_val = points_val * 1.1
        points_val = np.mean(num_points_ours_sorted[-1*int(ra):])
    
    new_dict[obj_key] = [mean_val, points_val]
    data_list.append([cnt, mean_val, points_val])

    name_list.append(obj_key)
    cnt += 1
data_list = np.array(data_list)
indices = data_list[:,0]
close_obj_array = data_list[:,1]
num_points_array = data_list[:,2]


arg = np.argsort(close_obj_array)
for idx in range(len(arg)):
    num_close_obj = close_obj_array[arg[idx]]
    name_idx = indices[arg[idx]]
    avg_num_points = num_points_array[arg[idx]]
    obj_name = name_list[int(name_idx)]
    
    print('{0} - close same obj: {1} avg num_points: {2}'.format(obj_name, num_close_obj, avg_num_points))
#pdb.set_trace()


#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
points_close = []
points_num = []
for interest_key in obj_interest:
    close_num, num_points = new_dict[interest_key]

    points_close.append(close_num)
    points_num.append(num_points)
x_labels = obj_interest
if x_labels[1] == 'shower curtain':
    x_labels[1] = 's. curtain'
x= np.arange(len(obj_interest))
y= points_close#[487, 420.8, 376.3, 240, 140, 160]#speed
num_points = points_num
#mAP = [56.5, 54.3, 58.4, 54.5, 47.3, 62.3] 

widths = [0.5,0.5,0.5,0.5,0.5,0.5]
colors = ["#a65628", "#ff7f00", "#984ea3", "#4daf4a", "#377eb8", "#e41a1c"]
fig, ax1 = plt.subplots()

ax1.set_xticks(x, x_labels)#, rotation='vertical')
ax1.bar(x,y, width=widths, color=colors)
ax1.set_ylim([0,2.5])
#pdb.set_trace()
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
ax2 = ax1.twinx()
ax2.plot(x, num_points, '--bo', color=(0,0,0), linewidth=3, label='Number of Obj. Points')

ax2.set_ylim([0,220000])
#ax2.tick_params(axis='y', which='major', labelsize=13)
#ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#ax2.set_ylabel('Number of Avg. Points', fontsize=15)
plt.yticks(fontsize=0)
plt.grid()
plt.legend(fontsize=12)
plt.savefig('vis/3_s3dis.png')
plt.clf()