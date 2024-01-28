import os
import numpy as np

def concatenate_features(folder_A, folder_B, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    log_file=open('concat.log','w')

    # 遍历文件夹A中的所有.npy文件
    for file_name in os.listdir(folder_A):
        if file_name.endswith('.npy'):
            # 构建文件路径
            file_path_A = os.path.join(folder_A, file_name)
            file_path_B = os.path.join(folder_B, file_name.replace('_clip','_videomae'))
            output_file_path = os.path.join(output_folder, file_name.replace('_clip','videomae'))
            if os.path.exists(output_file_path):
                continue

            # 加载文件A和B的特征数据
            features_A = np.load(file_path_A)
            features_B = np.load(file_path_B)
            if features_B.shape[1]<features_A.shape[1]:
                lastframe=features_B[:,-1,:]
                lastframe = np.expand_dims(lastframe, axis=1)

                features_B = np.concatenate((features_B, lastframe), axis=1)

            # 在C维度上进行拼接
            if features_B.shape[1]==features_A.shape[1]:
                concatenated_features = np.concatenate((features_A, features_B), axis=2)
                np.save(output_file_path, concatenated_features)
                print(f"Concatenated features saved to: {output_file_path}")
            else:
                log_file.write(f'A:{os.path.basename(file_path_A)}: {features_A.shape}  '
                               f'B:{os.path.basename(file_path_B)}: {features_B.shape} \n')

# 示例用法
# folder_A = '/home/sh/TEVAD/save/Crime/UCF_ten_crop_clip'
# folder_B = '/home/sh/TEVAD/save/Crime/9-5_9-1_finetune'
# output_folder = '/home/sh/TEVAD/save/Crime/Clip_AISO.5_concat'

folder_A = '/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_9-5_9-1_finetune'
folder_B = '/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_9-5_9-1_finetune_KMeans'
output_folder = '/home/sh/TEVAD/save/TAD/test'


concatenate_features(folder_A, folder_B, output_folder)