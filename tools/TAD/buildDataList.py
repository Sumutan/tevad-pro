import os
import pprint
import shutil

"""
buildDataList_TAD：
    通过替换.list文件中每行的路径前缀获得新的train/val/test set的所有文件绝对路径
"""


# 读取.list文件如lis_x264_videomae.npy，返回列表
def getlist(listpath):
    with open(listpath, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


# 将列表写入文件
def writelist(new_lines, listpath):
    with open(listpath, "w") as f:
        for line in new_lines:
            f.write(line + '\n')


def check_files_existence(file_list):
    """
    检查文件是否存在，返回不存在的文件列表
    如果所有文件存在，返回空列表
    """
    missing_files = []
    for file in file_list:
        if not os.path.isfile(file):
            missing_files.append(file)
    return missing_files


# 替换TAD文索引中的首尾
def rep_TAD(oldline: str, feature_name):
    newline = oldline.replace("/home/sh/TEVAD/save/TAD/TAD_9-5_9-1_finetune_dif_0.5/",
                              f"/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/{feature_name}/")
    return newline


def build_TAD_list(featureName, source_file_path, aim_file_path):
    # 获取列表文件
    source_file_list = getlist(source_file_path)
    # 处理列表
    feature = 'videomae'  # 'videomae'/'i3d'
    source_file_list = [rep_TAD(line, feature_name).replace('videomae',feature) for line in source_file_list]

    missing_files = check_files_existence(source_file_list)
    pprint.pprint(missing_files)
    print("miss files num:", len(missing_files))
    # 写出
    writelist(source_file_list, aim_file_path)


if __name__ == '__main__':
    feature_name = 'TAD_GL-20e-nofreeze'

    source_file_path_train = 'pathlist/TAD_train_list_AISO_0.5.txt'
    source_file_path_val = 'pathlist/TAD_val_list_AISO_0.5.txt'
    aim_file_path_train = f'TAD-videoMae-{feature_name.replace("TAD_", "")}.list'
    aim_file_path_val = f'TAD-videoMae-test-{feature_name.replace("TAD_", "")}.list'

    build_TAD_list(feature_name, source_file_path_train, aim_file_path_train)
    build_TAD_list(feature_name, source_file_path_val, aim_file_path_val)
