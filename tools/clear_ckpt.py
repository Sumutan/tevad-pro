"""
对ckpt进行清理，每个文件夹内仅保留最新的2个文件（best&final）
"""

import os
import glob
import datetime

# 指定文件夹路径
folder_path = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/ckpt/TAD-videoMae-GL-20e"

def clear_ckpt(folder_path,leave=2):
    '''剩余leave个最新的.pkl文件'''
    # 获取文件夹中所有.pkl文件的列表
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    # 按文件的修改时间进行排序，最新的文件排在前面
    sorted_files = sorted(pkl_files, key=os.path.getmtime, reverse=True)
    if len(sorted_files)>leave:
        # 保留最新的两个文件，删除其余文件
        for file in sorted_files[leave:]:
            os.remove(file)
            print(f"remove {file}")

root_folder = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/ckpt"
for folder_path, folder_name, file_names in os.walk(root_folder):
    for file_name in folder_name:
        folder = os.path.join(folder_path, file_name)
        clear_ckpt(folder)