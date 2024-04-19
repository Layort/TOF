import math
import cv2
import numpy as np
import json
import copy
from PIL import Image
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from skimage.transform._geometric import _umeyama as get_sym_mat
import torch.nn.functional as F
import torch

NEW_DATA_PATH = "new_data_file"


MIN_TEXT_AREA_RATIO = 0.02
BOUND_RATIO = 0.05

def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0]-pts1[1]), np.linalg.norm(pts1[2]-pts1[3]))
    height = max(np.linalg.norm(pts1[0]-pts1[3]), np.linalg.norm(pts1[1]-pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, :int(height), :int(width)]
    return result

def pre_process(img_in, shape):
    # rotate
    img = img_in
    h, w = img.shape[1:]
    if h > w * 1.2:
        img = torch.transpose(img, 1, 2).flip(dims=[1])
        img_in = img
        h, w = img.shape[1:]
    # resize
    imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
    assert imgC == img.shape[0]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=(imgH, resized_w),
        mode='bilinear',
        align_corners=True,
    )
    # padding
    padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
    padding_im[:, :, 0:resized_w] = resized_image[0]
    return padding_im.permute(1, 2, 0).cpu().numpy()  # HWC ,numpy
    

def main():
    # read preprocessed images
    file = open(NEW_DATA_PATH + "/img_name.txt", "r")
    img_name_in_lines = file.readlines()
    file.close()
    img_name_list = [i.strip() for i in img_name_in_lines]
    fliter_img_name_list =[]
    
    predictor = pipeline(Tasks.ocr_recognition, model="ocr_model")

    for img_name in tqdm(img_name_list):
        wordBB = np.load(NEW_DATA_PATH + "/wordBB/" + img_name  + ".npy")
        img = cv2.imread(NEW_DATA_PATH + "/img/" + img_name )
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()  # HWC-->CHW
        mask = Image.open(NEW_DATA_PATH + "/mask/" + img_name )
        mask_array = np.array(mask)/255
        W,H = mask_array.shape[0],mask_array.shape[1]

        txt_area_ratio = mask_array.sum()/(W*H)
        if(txt_area_ratio/wordBB.shape[2] > MIN_TEXT_AREA_RATIO):
            for i in range(wordBB.shape[2]):
                bb = wordBB[:,:,i]
                bb = bb.transpose()
                crop_img = adjust_image(bb,img)
                crop_img = pre_process(crop_img, "3, 48, 320")
                re = predictor(crop_img)
                if(re['text'][0] != ''): # if ocr model can recongnize at least one word
                    fliter_img_name_list.append(img_name)
                    # print(re['text'][0])
                    break;
    print(len(fliter_img_name_list),len(fliter_img_name_list)/len(img_name_list))
    file = open(NEW_DATA_PATH + "/fliter_img_name.txt", "w")
    for name in fliter_img_name_list:
        file.write(name+"\n")
    file.close()

def get_exclude():
    # read preprocessed images
    file = open(NEW_DATA_PATH + "/img_name.txt", "r")
    img_name_in_lines = file.readlines()
    file.close()
    img_name_list = {i.strip():1 for i in img_name_in_lines}
    file = open(NEW_DATA_PATH + "/fliter_img_name.txt", "r")
    img_name_in_lines = file.readlines()
    file.close()
    fliter_img_name_list ={i.strip():1 for i in img_name_in_lines}

    exclude_img_name_list = []
    for i in img_name_list.keys():
        if(i not in fliter_img_name_list):
            exclude_img_name_list.append(i)
    file = open(NEW_DATA_PATH + "/exclude_img_name.txt", "w")
    for name in exclude_img_name_list:
        file.write(name+"\n")
    file.close()

    print(len(exclude_img_name_list), len(fliter_img_name_list), len(exclude_img_name_list) + len(fliter_img_name_list),len(img_name_list) )

if __name__ == "__main__":
    main()