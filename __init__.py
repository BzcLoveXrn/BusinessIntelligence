import os


def print_directory_structure(root_dir):
    # 用于存储结果的字符串列表
    output = []

    for root, dirs, files in os.walk(root_dir):
        # 排除 .idea 文件夹和图片文件
        dirs[:] = [d for d in dirs if d not in ['.idea', 'venv','.git','.ipyb_checkpoints','__pycache__']]  # 排除文件夹
        files[:] = [f for f in files if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]  # 排除图片文件

        # 获取当前目录层级
        level = root.replace(root_dir, '').count(os.sep)
        indent = '  ' * level  # 使用 2 个空格作为缩进

        # 获取当前目录名称并添加到输出列表
        output.append(f"{indent}- **{os.path.basename(root)}/**")

        # 处理当前目录中的文件
        subindent = '  ' * (level + 1)
        for file in files:
            output.append(f"{subindent}- {file}")

    # 将输出拼接成一个字符串，返回给调用者
    return '\n'.join(output)


# 调用函数并打印结果
project_structure = print_directory_structure('.')
print(project_structure)
