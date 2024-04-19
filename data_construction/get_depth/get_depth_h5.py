# @misc{https://doi.org/10.48550/arxiv.2302.12288,
#   doi = {10.48550/ARXIV.2302.12288},
#   url = {https://arxiv.org/abs/2302.12288},
#   author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
#   keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
#   title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
#   publisher = {arXiv},
#   year = {2023},
#   copyright = {arXiv.org perpetual, non-exclusive license}
# }
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch
import os
import h5py
from zoedepth.utils.misc import colorize


# ZoeD_NK
conf = get_config("zoedepth_nk", "infer", pretrained_resource="local::/path/to/local/ckpt.pt")#  ZoeD_M12_NK 
model_zoe_nk = build_model(conf)


##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)


# Local file
from PIL import Image
root_path = '' #the filepath of the flickr30k images
ori_image = 'flickr30k_image' # the images
depth_file = 'depth.h5' # the depth maps with corresponding images
file_names = os.listdir(os.path.join(root_path,ori_image))
print(len(file_names))
print('file_path',os.path.join(root_path,ori_image))
output_depth = os.path.join(root_path,depth_file)
from  tqdm import tqdm
#create a h5 file and write the depth maps
with h5py.File(output_depth, 'w') as hf:
    for i in tqdm(range(len(file_names))):
        file_name = file_names[i]
        image = Image.open(os.path.join(root_path,ori_image,file_name)).convert("RGB")  # load
        depth = zoe.infer_pil(image)
        hf.create_dataset(file_name, data=depth)
        



