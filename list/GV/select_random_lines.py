import random

def select_random_lines(input_file, output_file, proportion=0.5):
    # 打开输入文件并读取所有行
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 计算需要选择的行数
    num_lines_to_select = int(len(lines) * proportion)

    # 获取所有行的索引
    indices = list(range(len(lines)))

    # 随机选择指定比例的索引，保持顺序
    selected_indices = sorted(random.sample(indices, num_lines_to_select))

    # 根据选中的索引获取对应的行
    selected_lines = [lines[i] for i in selected_indices]

    # 将选择的行写入到输出文件
    with open(output_file, 'w') as f:
        f.writelines(selected_lines)

# 使用示例
input_file = 'ucf-clip.list'   # 输入文件路径
output_file = 'ucf-clip-50%.list'  # 输出文件路径
proportion = 0.50  # 选择比例
select_random_lines(input_file, output_file, proportion)