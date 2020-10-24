import os
import torch
import time

"""
mkdir:
    如果 "path" 不存在则创建一个新文件夹.
"""
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

"""
write_doc:
    将 "content" 写入路径为 "path" 的 ".txt" 文档.
"""
def write_doc(path, content):
    with open(path, 'a') as file:
        file.write(content)

"""
get_time:
    获取当前时间.
"""
def get_time():
    torch.cuda.synchronize()
    return time.time()