import os
import numpy as np

"""
示例用法
folder_A = '/home/sh/TEVAD/save/Crime/UCF_ten_crop_clip'
folder_B = '/home/sh/TEVAD/save/Crime/9-5_9-1_finetune'
output_folder = '/home/sh/TEVAD/save/Crime/Clip_AISO.5_concat'
concatenate_features(folder_A, folder_B, output_folder)
"""

def concatenate_features(folder_A, folder_B, output_folder): #A is clip folder
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    log_file=open(f'{output_folder}/../concat_{folder_B.split("/")[-1]}.log','w')

    # 遍历文件夹A中的所有.npy文件
    for file_name in os.listdir(folder_A):
        if file_name.endswith('.npy'):
            # 构建文件路径
            file_path_A = os.path.join(folder_A, file_name)
            # file_path_B = os.path.join(folder_B, file_name.replace('_clip','_videomae')) #default
            # file_path_B = os.path.join(folder_B, file_name.replace('_clip_large','_videomae'))
            file_name = file_name.replace('.npy','_videomae.npy') #XD
            file_path_B = os.path.join(folder_B, file_name)#XD
            output_file_path = os.path.join(output_folder, file_name.replace('_clip','_videomae'))
            if os.path.exists(output_file_path):  #跳过已经完成的特征文件
                continue

            # 加载文件A和B的特征数据:由于不同的特征提取方法对于最后不足16frames的处理方法不同，
            # 提取得到的特征可能存在1段特征的长度差异，此处进行补足
            features_A = np.load(file_path_A)
            features_B = np.load(file_path_B)
            if features_B.shape[1]<features_A.shape[1]:
                lastframe=features_B[:,-1,:]
                lastframe = np.expand_dims(lastframe, axis=1)
                features_B = np.concatenate((features_B, lastframe), axis=1)
            elif features_A.shape[1]<features_B.shape[1]:
                lastframe=features_A[:,-1,:]
                lastframe = np.expand_dims(lastframe, axis=1)
                features_A = np.concatenate((features_A, lastframe), axis=1)

            # 在C维进行拼接
            if features_B.shape[1]==features_A.shape[1]:  # [crop,T,C]
                concatenated_features = np.concatenate((features_A, features_B), axis=2)
                np.save(output_file_path, concatenated_features)
                print(f"Concatenated features saved to: {output_file_path}")
            else: #输出错误日志
                log_file.write(f'A:{os.path.basename(file_path_A)}: {features_A.shape}  '
                               f'B:{os.path.basename(file_path_B)}: {features_B.shape} \n')

# CLIP_folder
# CLIP_folder = '/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Shanghai/ST_ten_crop_clip_add_512'
CLIP_folder = '/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Violence/10crop_clip_L'
# CLIP_folder = '/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Crime/UCF_ten_crop_clip'
# CLIP_folder = '/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/10crop_clip_large'
# CLIP_folder = '/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/TAD_10crop_clip'
# CLIP_folder = '/media/cw/584485FC4485DD5E/dataset/ucfcrime/clip_features_1crop'

#batch procssing
folders_B = [
    'XD_9-5_9-1_finetune_AISO_0.5',

]
dataset='Violence' #Crime/TAD/Shanghai/Violence

# output_folder = f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/{dataset}/CLIP-concat/'
output_folder = f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/{dataset}/CLIP-L-concat/'
# output_folder = f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Shanghai/CLIP-concat/'
# output_folder = f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Crime/CLIP-concat/'
# output_folder = f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/CLIP-concat/'

for folder_B in folders_B:
    folder_B_fullpath=f'/media/cw/584485FC4485DD5E/csh/tevad-pro/save/{dataset}/'+folder_B
    # folder_B_fullpath='/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/'+folder_B
    # folder_B_fullpath='/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Crime/'+folder_B
    # folder_B_fullpath='/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Shanghai/'+folder_B

    concatenate_features(CLIP_folder, folder_B_fullpath, output_folder+folder_B)
