from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
# from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid  
from io import BytesIO
import random

from tsv import TSVFile

from io import BytesIO
import base64
from PIL import Image
import numpy as np


def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')


def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr

def decode_item(item):
    item = json.loads(item)
    # print('item',item）
    item['image'] = decode_base64_to_pillow(item['image'])
   
    data_id = item['data_id']
    file_name = item['file_name']
    file_path = os.path.join('save_file',file_name)
    item['image'].save(file_path, "JPEG")


if __name__ == "__main__":
    tsv_file = TSVFile('tsv.file')
    print('len', len(tsv_file))
    for i in range(len(tsv_file)):
        _, item = tsv_file[i]
        item = decode_item(item)