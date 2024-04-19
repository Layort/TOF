import torch 
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision
from zipfile import ZipFile 
import os
import multiprocessing
import math
import numpy as np
import random 


VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png']


def check_filenames_in_zipdata(filenames, ziproot):
    samples = []
    for fst in ZipFile(ziproot).infolist():
        fname = fst.filename
        if fname.endswith('/') or fname.startswith('.') or fst.file_size == 0:
            continue
        if os.path.splitext(fname)[1].lower() in VALID_IMAGE_TYPES:
            samples.append((fname))
    filenames = set(filenames)
    samples = set(samples)
    assert filenames.issubset(samples), 'Something wrong with your zip data'



def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline =colors[bid % len(colors)], width=2)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 



def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    valid = True

    if x0>image_size or y0>image_size or x1<0 or y1<0:
        valid = False # no way to make this box vide, it is completely cropped out 
        return valid, (None, None, None, None)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)

    if (x1-x0)*(y1-y0) / (image_size*image_size) < min_box_size:
        valid = False
        return valid, (None, None, None, None)
     
    return valid, (x0, y0, x1, y1)





def recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, image_size, min_box_size):
    """
    x,y,w,h:  the original annotation corresponding to the raw image size.
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """

   
    x0 = x * trans_info["performed_scale_x"] - trans_info['crop_x'] 
    y0 = y * trans_info["performed_scale_y"] - trans_info['crop_y'] 
    x1 = (x + w) * trans_info["performed_scale_x"] - trans_info['crop_x'] 
    y1 = (y + h) * trans_info["performed_scale_y"] - trans_info['crop_y'] 

    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)

    if valid:
        # we also perform random flip. 
        # Here boxes are valid, and are based on image_size 
        if trans_info["performed_flip"]:
            x0, x1 = image_size-x1, image_size-x0

    return valid, (x0, y0, x1, y1)


def recalculate_box_and_verify_if_valid2(bbox, trans_info, image_size, min_box_size):
    X = bbox[0,:]
    Y = bbox[1,:]
    X = X * trans_info["performed_scale_x"] - trans_info['crop_x'] 
    Y = Y * trans_info["performed_scale_y"] - trans_info['crop_y'] 
    if((X<0).any() or (X >image_size).any() or (Y<0).any() or (Y> image_size).any() ):
        valid = False # no way to make this box vide, it is completely cropped out 
        return valid, None
    x0 = min(X); x1 = max(X)
    y0 = min(Y); y1 = max(Y)
    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)
    if(not valid):
        return valid, None;
    return valid, np.array([X,Y])

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, random_crop, random_flip, image_size):
        super().__init__() 
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.image_size = image_size
        self.zip_dict = {}

        if self.random_crop:
            assert False, 'NOT IMPLEMENTED'


    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file


    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_caption=True):
    
        if out is None:
            out = self[index]

        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
        canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        W, H = img.size

        if print_caption:
            caption = out["caption"]
            print('caption',caption)
            print(" ")

        boxes = []
        for box in out["boxes"]:    
            x0,y0,x1,y1 = box
            boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )

        img = draw_box(img, boxes )
        
        if return_tensor:
            return  torchvision.transforms.functional.to_tensor(img)
        else:
            img.save(name)   


    def transform_image(self, pil_image):
        if self.random_crop:
            assert False
            arr = random_crop_arr(pil_image, self.image_size) 
        else:
            arr, info = center_crop_arr(pil_image, self.image_size)

        # print('arr_image.shape',arr.shape)
         
        info["performed_flip"] = False
        arr = arr.astype(np.float32) / 127.5 - 1  # [-1, 1]
        arr = np.transpose(arr, [2,0,1])

        # print('arr_image_shape',arr.shape)
        
        # return torch.tensor(arr)
        return torch.tensor(arr), info 

    def transform_mask_image(self,mask_image):
        if self.random_crop:
            assert False
            arr = random_crop_arr(pil_image, self.image_size) 
        else:
            arr, info = center_crop_arr(mask_image, self.image_size)
        
        info["performed_flip"] = False
        
        arr = 1 - arr.astype(np.float32) / 255.0   # [0, 1]
        arr = np.transpose(arr, [2,0,1])

        return torch.tensor(arr), info 

def center_crop_arr(pil_image, image_size):

    WW, HH = pil_image.size

    scale_x = image_size/WW
    scale_y = image_size/HH
    pil_image = pil_image.resize(
        (image_size,image_size), resample=Image.BICUBIC
    )
    crop_y = 0
    crop_x = 0

    arr = np.array(pil_image)

    info = {"performed_scale_x":scale_x, "performed_scale_y":scale_y, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH}

    return arr, info



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
