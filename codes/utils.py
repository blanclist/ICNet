import os
import torch
import time

"""
mkdir:
    Create a folder if "path" does not exist.
"""
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

"""
write_doc:
    Write "content" into the file(".txt") in "path".
"""
def write_doc(path, content):
    with open(path, 'a') as file:
        file.write(content)

"""
get_time:
    Obtain the current time.
"""
def get_time():
    torch.cuda.synchronize()
    return time.time()