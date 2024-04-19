# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as describe
d in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""
import os
from synthgen import *
from common import *
from functools import reduce
import re
from time import time
from data_provider import DateProvider
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform._geometric import _umeyama as get_sym_mat

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
# Define some configuration variables:
NUM_IMG = -1  # number of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 8  # number of times to use the same image
SECS_PER_IMG = 2  # max time per image in seconds
NUM_GENED = 0
MIN_TEXT_AREA_RATIO = 0.05

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'Flickr30K file'
MASKS_DIR = "mask file"
SAVE_DIR = "new image save file"

RATIO = 0.72
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
    


def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        img_id = imgname.split(".")[0] 
        dname = "%s_%d.jpg" % (img_id, i)  
        print(dname) 
        #  concentrate masks
        masks_merged = reduce(lambda a, b: np.add(a, b), res[i]['masks']) 
        masks_merged = np.divide(masks_merged, len(res[i]['masks'])) 
        shape = res[i]['img'].shape 
        group = np.zeros((shape[0],shape[1],shape[2] + 1) , dtype = "uint8") #add by 
        group[:,:,:3] = res[i]['img'] 
        group[:,:,3] = masks_merged 

        db['data'].create_dataset(dname, data=group) # change by 
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        L = res[i]['txt']
        L = [n.encode("ascii", "ignore") for n in L]
        db['data'][dname].attrs['txt'] = L
        db['data'][dname].attrs['fg_color'] = res[i]['fg_color'] 

def save_to_path(predictor,imgname, res,img_name_list):
    """
    Save data
    """
    global NUM_GENED
    ninstance = len(res)
    success_number = 0  #这一参数在生成test的时候使用 ，train不用
    for i in range(ninstance):
        img_id = imgname.split(".")[0] 
        dname = "%s_%d.jpg" % (img_id, i)  
        img_name_list.append(dname)
        #  合并masks
        masks_merged = reduce(lambda a, b: np.add(a, b), res[i]['masks']) 
        masks_merged = np.divide(masks_merged, len(res[i]['masks'])) 
        img = res[i]['img']
        img_ocr = torch.from_numpy(img)
        img_ocr = img_ocr.permute(2, 0, 1).float()  # HWC-->CHW
        charBB = res[i]['charBB']
        wordBB = res[i]['wordBB']
        W,H = masks_merged.shape[0],masks_merged.shape[1]
        txt_area_ratio = masks_merged.sum()/(W*H)
        L = res[i]['txt']
        L = [n.encode("ascii", "ignore") for n in L]
        txt = res[i]['txt']
        fg_color = res[i]['fg_color'] 
        txt_old = txt
        txt = []
        for i in txt_old:
            for j in(i.split()):
                txt.append(j)
        assert len(txt) == wordBB.shape[2], f"{txt,wordBB.shape} not match in shape"
        
        if(txt_area_ratio/wordBB.shape[2] <= MIN_TEXT_AREA_RATIO):
            continue
        else:
            for i in range(wordBB.shape[2]):
                bb = wordBB[:,:,i]
                bb = bb.transpose()
                crop_img_ocr = adjust_image(bb,img_ocr)
                crop_img_ocr = pre_process(crop_img_ocr, "3, 48, 320")
                re = predictor(crop_img_ocr)
                
                if(re['text'][0] != '' and success_number<=3):  #这一参数在生成test的时候使用 ，train不用
                    success_number += 1
                    NUM_GENED += 1
                    plt.imsave(SAVE_DIR + "/img/" + dname,img)
                    imageio.imwrite(SAVE_DIR + "/mask/" + dname, masks_merged) 
                    np.save(SAVE_DIR + "/charBB/"+dname + ".npy",charBB)
                    np.save(SAVE_DIR + "/wordBB/"+dname + ".npy",wordBB)
                    file = open(SAVE_DIR + "/txt/" + dname  + ".txt","w")
                    file.write(str(txt))
                    np.save(SAVE_DIR + "/fg_color/"+dname + ".npy",fg_color)
                    file.close()



def main(viz=False, debug=False, output_masks=False, data_path=None):
    """
    Entry point.

    Args:
        viz: display generated images. If this flag is true, needs user input to continue with every loop iteration.
        output_masks: output masks of text, which was used during generation
    """
    if output_masks:
        # create a directory if not exists for masks
        if not os.path.exists(MASKS_DIR):
            os.makedirs(MASKS_DIR)
    if not (os.path.exists(SAVE_DIR)):
        os.makedirs(SAVE_DIR)
    if not (os.path.exists(SAVE_DIR + "/img")):
        os.makedirs(SAVE_DIR+ "/img")
    if not (os.path.exists(SAVE_DIR+ "/charBB")):
        os.makedirs(SAVE_DIR+ "/charBB")
    if not (os.path.exists(SAVE_DIR+ "/wordBB")):
        os.makedirs(SAVE_DIR+ "/wordBB")
    if not (os.path.exists(SAVE_DIR+ "/txt")):
        os.makedirs(SAVE_DIR+ "/txt")
    if not (os.path.exists(SAVE_DIR+ "/mask")):
        os.makedirs(SAVE_DIR+ "/mask")
    if not (os.path.exists(SAVE_DIR+ "/fg_color")):
        os.makedirs(SAVE_DIR+ "/fg_color")

    # open databases:
    
   
    provider = DateProvider(data_path) 
    predictor = pipeline(Tasks.ocr_recognition, model="orc_model")

    # get the names of the image files in the dataset:
    imnames = provider.get_imnames() # e.g. imnames = 'ant+hill_1.jpg'
    N = len(imnames)
    if debug:
        print(colorize(Color.BLUE, 'getting data..', bold=True))
        print(colorize(Color.BLUE, '\t-> done', bold=True))
        print('N',N)
    global NUM_IMG,NUM_GENED
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, int(N*RATIO))
    end_idx = int(end_idx)
    renderer = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)

    img_name_list= []
    for i in tqdm(range(start_idx, end_idx )):
        imname = imnames[i]

        try:
            # get the image:
            print('imname',imname)
            img = provider.get_image(imname)
            
            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            depth = provider.get_depth(imname)
        
           
            # get segmentation:
            seg = provider.get_segmap(imname)[:].astype('float32')
            # print('seg',seg)
        
            area = provider.get_segmap(imname).attrs['area']  # number of pixels in each region
            label = provider.get_segmap(imname).attrs['label'] 

            # re-size uniformly:  
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.LANCZOS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
            
 
            if debug:
                print('imname',imname)
                print('img',img.size)
                print('depth.shape',depth.shape)
                print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
                # print('sz',sz)
                print('img.shape',img.shape)
                print('seg.shape',seg.shape)
                print("\n    Processing " + str(imname) + "...")

            res = renderer.render_text(img, depth, seg, area, label, imname,
                                  ninstance=INSTANCE_PER_IMAGE)
            
            if len(res) > 0:
                # non-empty : successful in placing text:
                # add_res_to_db(imname, res, out_db)
                save_to_path(predictor,imname,res,img_name_list)
                # add_res_to_db(imname, res, out_db)
                if debug:
                    print("    Success. " + str(len(res[0]['txt'])) + " texts placed:")
                    print("    Texts:" + ";".join(res[0]['txt']) + "")
                    ws = re.sub(' +', ' ', (" ".join(res[0]['txt']).replace("\n", " "))).strip().split(" ")
                    print("    Words: #" +str(len(ws)) + " " + ";".join(ws) + "")
                    print("    Words bounding boxes: " + str(res[0]['wordBB'].shape) + "")
            else:
                # print("    Failure: No text placed.")
                pass

            if len(res) > 0 and output_masks:
                ts = str(int(time() * 1000))

                # executed only if --output-masks flag is set
                prefix = MASKS_DIR + "/" + imname + ts

                imageio.imwrite(prefix + "_original.png", img)
                imageio.imwrite(prefix + "_with_text.png", res[0]['img'])

                # merge masks together:
                merged = reduce(lambda a, b: np.add(a, b), res[0]['masks'])
                # since we just added values of pixels, need to bring it back to 0..255 range.
                merged = np.divide(merged, len(res[0]['masks']))
                imageio.imwrite(prefix + "_mask.png", merged)

                # print bounding boxes
                f = open(prefix + "_bb.txt", "w+")
                bbs = res[0]['wordBB']
                boxes = np.swapaxes(bbs, 2, 0)
                words = re.sub(' +', ' ', ' '.join(res[0]['txt']).replace("\n", " ")).strip().split(" ")
                assert len(boxes) == len(words)
                for j in range(len(boxes)):
                    as_strings = np.char.mod('%f', boxes[j].flatten())
                    f.write(",".join(as_strings) + "," + words[j] + "\n")
                f.close()

            # visualize the output:
            if viz:
                # executed only if --viz flag is set
                for idict in res:
                    img_with_text = idict['img']
                    viz_textbb(1, img_with_text, [idict['wordBB']], alpha=1.0)
                    viz_masks(2, img_with_text, seg, depth, idict['labeled_region'])
                    # viz_regions(rgb.copy(),xyz,seg,regions['coeff'],regions['label'])
                    if i < INSTANCE_PER_IMAGE - 1:
                        input(colorize(Color.BLUE, 'continue?', True))
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
        # exit(0)
    provider.close()
    file = open(SAVE_DIR + "/img_name.txt","w")
    for i in img_name_list:
        file.write(i)
        file.write("\n")
    file.close()
    print("gen %d"%(NUM_GENED))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--output-masks', action='store_true', dest='output_masks', default=False,
                        help='flag for turning on output of masks')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help='flag for turning on debug output')
    parser.add_argument("--data", type=str, dest='data_path', default='bg_data_20k/',
                        help="absolute path to data directory containing images, segmaps and depths")
    args = parser.parse_args()
    main(viz=args.viz, debug=args.debug, output_masks=args.output_masks, data_path=args.data_path)
