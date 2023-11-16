import numpy as np
import os

def concatenate_npy_files(folder_path, feature_name):
    # 获取文件夹下的.npy文件列表
    file_list = [file for file in os.listdir(folder_path) if (feature_name in file and file.endswith(".npy"))]

    # 根据文件名中的数字进行排序
    file_list.sort(key=lambda x: int(x.split("_x264_")[1].split("_")[0]))

    # 读取.npy文件并拼接在T维度上
    arrays = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        array = np.load(file_path)
        arrays.append(array)

    # 在T维度上进行拼接
    result = np.concatenate(arrays, axis=1)

    # 保存拼接后的结果为.npy文件
    new_name=feature_name+'_videomae.npy'
    output_folder=os.path.join(folder_path,'concated')
    output_path = os.path.join(output_folder ,new_name)
    os.makedirs(output_folder,exist_ok=True)
    np.save(output_path, result)

    print("拼接完成并保存为", new_name)

# 测试函数
folder_path = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/tmp/UCF_9-5_9-1_finetune_dif_0.5_SP_norm_clip"  # 替换为实际的文件夹路径
feature_names = ["Normal_Videos307_x264","Normal_Videos308_x264","Normal_Videos633_x264.npy"]
for f in feature_names:
    concatenate_npy_files(folder_path, f)
##记得手动拖拽下5xx!!