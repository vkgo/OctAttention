# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: copyPlyFiles.py
# @Author: vkgo
# @E-mail: hwjho@qq.com, csvk@mail.scut.edu.cn
# @Time: Apr 21, 2023
# ---
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_ply_file(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
        logging.info(f'复制文件: {src_path} -> {dst_path}')
    except Exception as e:
        logging.error(f'复制文件时出错: {src_path} -> {dst_path}，错误信息：{e}')

def find_and_copy_ply_files(folder, dst_folder, executor):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ply'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)
                executor.submit(copy_ply_file, src_path, dst_path)

# 指定源文件夹路径数组
src_folders = [
    './8iVFBv2',
    './8iVLSF_910bit',
]

# 指定目标文件夹路径
dst_folder = './testplyfiles'

# 检查目标文件夹是否存在，如果不存在，则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 创建线程池
with ThreadPoolExecutor() as executor:
    # 遍历源文件夹路径数组，提交任务到线程池
    for folder in src_folders:
        find_and_copy_ply_files(folder, dst_folder, executor)
