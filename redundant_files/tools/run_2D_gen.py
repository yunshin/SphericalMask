import sys
import os 
import pdb


scans_path = '/root/src/SoftGroup/dataset/scannetv2/scans_test'
out_path = '/root/nas/madhu/data/scannet_test'
scan_names = os.listdir(scans_path)
for scan in scan_names:

    file_path = os.path.join(scans_path,scan, scan+'.sens')
    output_path = os.path.join(out_path, scan)
    
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    command = 'python transfer_data_madhu.py --filename {0} --output_path {1} --export_depth_images --export_color_images --export_poses --export_intrinsics'.format(file_path, output_path)
    print(command)
    os.system(command)
#files = 
#python transfer_data_madhu.py --filename /root/src/SoftGroup/dataset/scannetv2/scans_test/scene0707_00/scene0707_00.sens --output_path /root/nas/madhu/data/scannet_test --export_depth_images --export_color_images --export_poses --export_intrinsics

