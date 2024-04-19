import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
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

def get_item(item):
    item = json.loads(item)
    data_id = item['data_id']
    file_name = item['file_name']
    caption = item['caption']
    annos = [{key: value for key, value in d.items() if key != "text_embedding_before" and key != "text_embedding_after" and key != "image_embedding_before" and key != "image_embedding_after"} for d in item['annos']]

    return data_id, file_name, caption, annos



def decode_item(item):
    item = json.loads(item)
    # print('item',item)

    item['image'] = decode_base64_to_pillow(item['image'])
    data_id = item['data_id']
    print('data_id',data_id)
    file_name = item['file_name']
    file_path = os.path.join('save_file',f'{data_id:06d}'+'_'+file_name)
    item['image'].save(file_path, "JPEG")
    print('save image:'+f'{data_id}'+'\n')
    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item


if __name__ == "__main__":
    tsv_file = TSVFile('tsv_file')
    print('len', len(tsv_file))
    images = []
    annotations = []
    for i in range(len(tsv_file)):
        id, item = tsv_file[i]
        image = {}
        data_id, file_name, caption, annos = get_item(item)
        image['id'] = data_id
        image['file_name'] = file_name
        image['caption'] = caption
        images.append(image)
        annotations.extend(annos)
        if i %10000 == 0:
            print('finished the list of %d',i)
    my_dict = {
        "info": {
        "dataset_name": "flickr30k_3359636318",
        "creation_date": "2023-09-11"
        },
        "images":images,
        "licenses": [
        {
            "license_id": 1,
            "type": "CC BY-SA 4.0",
            "holder": "John Doe"
        }
        ],
        "annotations":annotations
    with open('flickr30k.json','w',encoding='utf-8') as fp: #这里使用的是将数据转化成json，对文件进行命名‘data.json’，或者其他
        json.dump(my_dict,fp) #使用json.dump()函数，第一个参数是列表数据，第二个参数似乎是指针指向fp（即'data.json'）
