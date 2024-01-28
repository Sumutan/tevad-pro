"""
批量修改目标文件夹下的所有文件:
.npy 文件的shape从[crop,t,1,C]->[crop,t,C]
"""
import os
import numpy as np

def rename(file_name):
    if file.endswith('.npy'):
        new_name=file_name.replace('.mp4','')
        os.renames(file_name,new_name)

def reshape(file_name):  # file_name:real_path of npy
    if file.endswith('.npy'):
        data = np.load(file_name)
        # 修改np文件的shape
        data = np.squeeze(data)  # 去除维度为1的维度
        print(data.shape)
        np.save(file_name, data)

dir = "/home/sh/TEVAD/save/TAD/TAD_9-5_9-1_finetune_KMeans_8centers"
for file in os.listdir(dir):
    file_name = os.path.join(dir, file)

    # rename(file_name)
    reshape(file_name)


