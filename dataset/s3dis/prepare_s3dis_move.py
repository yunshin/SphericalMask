import numpy as np
import torch
from scipy.spatial import KDTree

import configargparse
import glob
import natsort
import os
import sys
import pdb

sys.path.append(".")


save_dir = "/root/src/ISBNet/dataset/s3dis/preprocess/"

files = os.listdir(save_dir)

for f_name in files:


    if '_inst_nostuff' in f_name:
       
        os.system('mv {0} {1}'.format(save_dir+f_name, save_dir+f_name+'.pth'))
        print('mv {0} {1}'.format(save_dir+f_name, save_dir+f_name+'.pth'))