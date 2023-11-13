import os


def readtxt(file_path):  # 文本文件路径
    with open(file_path, "r") as file:  # 打开文本文件进行读取
        lines = file.readlines()
    lines = [line.strip() for line in lines]  # 去除每行末尾的换行符
    return lines


def parseline(line: str):
    fileName = line.split('.mp4')[0]
    return fileName


train_label = "/home/sh/TEVAD/save/TAD/label/trains.txt"
val_label = "/home/sh/TEVAD/save/TAD/label/vals.txt"

train_lines = readtxt(train_label)
val_lines = readtxt(val_label)

train_list = [parseline(line) for line in train_lines]
val_list = [parseline(line) for line in val_lines]

allFile_list = list(set(train_list) | set(val_list))


def delete_fullPath_file():
    # 删除多余文件
    folder = "/home/sh/TEVAD/save/TAD/TAD_9-5_9-1_finetune_dif_0.5"  # 替换为要遍历的文件夹路径
    for root, directories, files in os.walk(folder):
        print("len(files):", len(files))
        deleteList = []
        for file in files:
            filename = file.split('.mp4')[0]
            if filename not in allFile_list:
                deleteList.append(filename)
                try:
                    fullPath = os.path.join(root, file)
                    os.remove(fullPath)
                    print(f"成功删除文件: {fullPath}")
                except OSError as e:
                    print(f"删除文件时出错: {e}")
        print("len(deleteList):", len(deleteList))
        print(deleteList)


def file_name_process():
    folder = "/home/sh/TEVAD/save/TAD/TAD_9-5_9-1_finetune"  # 替换为要遍历的文件夹路径
    for root, directories, files in os.walk(folder):
        print("len(files):", len(files))
        for file in files:
            newFileName = file.replace(".mp4", "")
            try:
                fullPath = os.path.join(root, file)
                fullPath_new = os.path.join(root, newFileName)
                os.rename(fullPath, fullPath_new)
                print(f"成功将文件名从 {fullPath} 修改为 {fullPath_new}")
            except OSError as e:
                print(f"修改文件名时出错: {e}")

def write_PathList():
    head="/home/sh/TEVAD/save/TAD/TAD_9-5_9-1_finetune_dif_0.5"
    filelist=[]
    for name in val_list:  #train_list/val_list
        file=name+'_videomae.npy'
        line = os.path.join(head,file)
        filelist.append(line)
    # 将列表逐行写入文本文件
    with open("TAD_val_list.txt", "w") as file:
        for item in filelist:
            file.write(item + "\n")

file_name_process()