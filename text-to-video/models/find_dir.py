import os

def find_directory(root_dir, target_dir_name):
    cont = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        cont += 1
        print(f'cont:{cont}')
        if target_dir_name in dirnames:
            target_path = os.path.join(dirpath, target_dir_name)
            return target_path
    return None

root_directory = '/mnt/petrelfs/share_data'  # 修改为你的根目录路径
target_directory_name = 'Latte-0'

result = find_directory(root_directory, target_directory_name)

if result:
    print(f"Found directory: {result}")
else:
    print(f"Directory '{target_directory_name}' not found in '{root_directory}'")