"""
编写python代码，给定一个文件路径，对其中所有文件夹进行以下操作：
1.找到并打开其中以“test.log”结尾的数据
2.该数据中每一行都记着形如以下形式的
日志：“Epoch 530  rec_auc_all: 0.8245  rec_auc_abn: 0.8270  far_all: 0.0949 far_abn:0.0923”，
请找出所有行中“rec_auc_all”后数值最高的一行，记录保存在同级文件夹文件“best.log”下
"""


import os

def find_best_log(file_path):
    # 获取文件夹下的所有文件夹列表
    folder_list = [folder for folder in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, folder))]

    # 遍历文件夹列表
    for folder in folder_list:
        folder_path = os.path.join(file_path, folder)

        # 获取文件夹下以"test.log"结尾的文件列表
        file_list = [file for file in os.listdir(folder_path) if file.endswith("test.log")]

        if len(file_list) == 0:
            print(f"文件夹 {folder} 中不存在以 'test.log' 结尾的文件")
        else:
            best_line = None
            best_rec_auc_all = float('-inf')

            # 遍历文件列表，找到最高的 rec_auc_all 值
            for file in file_list:
                output_file_path = os.path.join(folder_path, file)
                with open(output_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "rec_auc_all" in line:
                            rec_auc_all = float(line.split("rec_auc_all:")[-1].split()[0])
                            if rec_auc_all > best_rec_auc_all:
                                best_rec_auc_all = rec_auc_all
                                best_line = line.strip()

            # 将最高的 rec_auc_all 值保存到 "best.log" 文件中
            if best_line:
                best_log_file = os.path.join(folder_path, f"best_AUC_{str(best_rec_auc_all)}.log")
                with open(best_log_file, 'w') as f:
                    f.write(best_line)
                print(f"文件夹 {folder} 中的最佳日志已保存到 {best_log_file}")
            else:
                print(f"文件夹 {folder} 中没有找到符合条件的日志行")

# 测试函数
# folder_path = "../output"  # 替换为实际的文件夹路径
# find_best_log(folder_path)

def get_subfolder_paths(directory):
    subfolder_paths = [directory]

    # 使用 os.walk 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            # 拼接当前子文件夹的绝对路径
            subfolder_path = os.path.join(root, dir_name)
            subfolder_paths.append(subfolder_path)

    return subfolder_paths

def delete_file_if_exists(directory):
    file_path = os.path.join(directory, "best.log")

    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"已删除文件: {file_path}")
    else:
        print(f"文件不存在: {file_path}")


# 示例用法
directory = "../output"
subfolders = get_subfolder_paths(directory)

# 对包括子文件夹的所有文件夹进行操作
for subfolder in subfolders:
    print(subfolder)
    find_best_log(subfolder)