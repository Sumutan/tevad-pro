import os

dir="/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_11-1_10-15_finetune_L"
for file in os.listdir(dir):
    old_name=os.path.join(dir,file)
    new_name=os.path.join(dir,file.replace(".mp4",""))

    os.renames(old_name,new_name)