import os


def replaceStr(source_list, replaced_str: str, aim_str: str, ):
    source_list = [filename.replace(replaced_str, aim_str) for filename in source_list]
    return source_list


# 定义输入文件和输出文件的路径
# input_file = 'src/shanghai-i3d-test-10crop.list'
# output_file = 'shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-test-10crop.list'
input_file = 'src/shanghai-i3d-train-10crop.list'
output_file = 'shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-train-10crop.list'

# 读取输入文件中的文本到列表中去
with open(input_file, 'r') as f:
    lines = f.readlines()
# i3d
# list_new=replaceStr(lines,'/media/lizi/新加卷/sh/TEVAD-main/save/Shanghai/SH_ten_crop_i3d_v2/',
#            '/home/sh/sh/programs/TEVAD-main/save/Shanghai/SH_ten_crop_i3d_v2/')
# list_new=replaceStr(list_new,'_i3d.npy','_i3d.npy')
# videoMAE
list_new = replaceStr(lines, '/media/lizi/新加卷/sh/TEVAD-main/save/Shanghai/SH_ten_crop_i3d_v2/',
                      '/home/sh/sh/programs/TEVAD-main/save/Shanghai/SHT_9-5_9-1_finetune/')
list_new = replaceStr(list_new, '_i3d.npy', '_videomae.npy')

# 将新的列表保存到输出文件中去
with open(output_file, 'w') as f:
    for line_new in list_new:
        f.write(line_new)
