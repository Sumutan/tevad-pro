import os
import pprint
import shutil

"""
buildDataList_ucf：
1.通过写有绝对路径的.list文件获得train/val/test集的所有文件绝对路径，
    并去掉首尾仅保留文件名存储在文件名列表中
2.为文件名列表中所有行添加合适的前缀和后缀，生成新的.list，保存文件
"""


# 读取.list文件如lis_x264_videomae.npy，返回列表
def getlist(listpath):
    with open(listpath, 'r') as f:
        lines = f.readlines()
    return lines


# 将列表写入文件
def writelist(new_lines, listpath):
    with open(listpath, "w") as f:
        for line in new_lines:
            f.write(line+'\n')


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


feature_name = 'UCF_ten_crop_videomae'
# 替换ucf文索引中的首尾
def rep_ucf(oldpath):
    new_prefix_ = f'/home/sh/sh/programs/TEVAD-main/save/Crime/{feature_name}/'  # 根据新的路径修改此处,记得加'/'
    new_suffix = f'_x264_videomae.npy' # _x264_i3d.npy/_x264_videomae.npy/_x264_clip.npy
    filename = oldpath.split('/')[-1].split('_x264')[0]  # ...SH_ten_crop_i3d_v2/01_0015_i3d.npy ->Abuse028
    return new_prefix_ + filename + new_suffix


if __name__ == '__main__':
    source_file_path = 'pathlist/ucf-i3d_raw.list'
    aim_file_path = f'ucf-videoMae-{feature_name}.list'
    # source_file_path = 'pathlist/ucf-i3d-test_raw.list'
    # aim_file_path = f'ucf-videoMae-test-{feature_name}.list'
    # 获取列表文件
    source_file_list = getlist(source_file_path)
    # 处理列表
    source_file_list = [rep_ucf(line) for line in source_file_list]

    missing_files=check_files_existence(source_file_list)
    pprint.pprint(missing_files)
    print("miss files num:",len(missing_files))

    # 写出
    writelist(source_file_list, aim_file_path)
