import os


def write_sorted_npy_paths_to_txt(directory, output_txt_file):
    # 用于存储所有.npy文件的绝对路径
    npy_file_paths = []

    # 遍历目录及子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为.npy文件
            if file.endswith('.npy'):
                # 获取文件的绝对路径
                file_path = os.path.join(root, file)
                # 将路径添加到列表中
                npy_file_paths.append(file_path)

    # 按照文件名（数字）进行排序
    npy_file_paths_sorted = sorted(npy_file_paths, key=lambda x: int(os.path.basename(x).split('_clip.npy')[0]))

    # 打开输出的文本文件
    with open(output_txt_file, 'w') as f:
        # 将排序后的路径写入文件
        for path in npy_file_paths_sorted:
            f.write(path + '\n')

    print(f"All .npy file paths have been written to {output_txt_file} in sorted order.")


# 示例用法
directory = '/home/cw/下载/descriptions_1_Features'  # 替换为你想遍历的文件夹路径
output_txt_file = 'npy_file_paths.txt'  # 输出的.txt文件路径

write_sorted_npy_paths_to_txt(directory, output_txt_file)


