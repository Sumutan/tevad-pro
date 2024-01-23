"""
批量修改目标文件夹下的所有文件名
"""
import os

dir="/home/sh/TEVAD/save/TAD/TAD_GL-20e-nofreeze"
for file in os.listdir(dir):
    old_name=os.path.join(dir,file)
    new_name=os.path.join(dir,file.replace(".mp4",""))

    os.renames(old_name,new_name)