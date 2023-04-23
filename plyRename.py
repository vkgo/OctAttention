import os
from pathlib import Path
import concurrent.futures

def rename_file(parent_folder, file, fontname):
    file_path = os.path.join(parent_folder, file)
    if os.path.isfile(file_path):
        new_file_name = f"{fontname}_{file}"
        new_file_path = os.path.join(parent_folder, new_file_name)
        os.rename(file_path, new_file_path)
    elif os.path.isdir(file_path):
        rename_files_in_folder(file_path)

def rename_files_in_folder(folder_path, fontname):
    parent_folder = Path(folder_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda file: rename_file(parent_folder, file, fontname), os.listdir(parent_folder))

if __name__ == "__main__":
    target_folder = './MVUB/sarah10/ply'
    fontname = 'sarah10'
    rename_files_in_folder(target_folder, fontname)
    print("文件重命名完成！")
