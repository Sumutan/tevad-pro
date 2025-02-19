"""
批量修改目标文件夹下的所有文件名
"""
import os

dir="/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/TAD_finetune_pretrain_with_RandomMaskinK400-100_457"
for file in os.listdir(dir):
    old_name=os.path.join(dir,file)
    new_name=os.path.join(dir,file.replace(".mp4",""))

    os.renames(old_name,new_name)