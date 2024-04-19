import h5py
from PIL import Image
import numpy as np
import cv2
import pickle
import os
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
# import torch



def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    # if isinstance(value, torch.Tensor):
    #     value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
def load_cp(file_path):
    
    with open(file_path, 'rb') as f:
        filtered_imnames = set(pickle.load(f))
    # 在这里，您可以对加载的数据进行任何操作，例如打印、分析等
    print(len(filtered_imnames))
    print(filtered_imnames)

    #请注意，如果您打开 .cp 文件并发现它是一个Pickle文件，那么您应该确保对从该文件加载的数据进行适当的处理，以便与您的程序的上下文相匹配。Pickle文件可能包含各种Python对象，从简单的数据结构到复杂的对象，具体内容取决于文件创建时使用的Python对象。
def read_dset_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        print("HDF5文件信息:")
        # print("文件名:", f.filename)
        # print("格式版本:", f.libver)
        # print("文件模式:", f.mode)
        # print(len(f))
        print("\n根级组:")
        for group_name in f.keys():
            group = f[group_name]
            print("组名:", group_name)
            print("  类型:", type(group))
            print("  ID:", group.id)
            print("  属性数量:", len(group.attrs))
            print("  数据集数量:", len(group))
            print(group)
            if group_name=='depth':
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    print("    数据集名:", dataset_name)
                    print("    数据集形状:", dataset.shape)
                    print("    数据集数据类型:", dataset.dtype)
                    dataset_name_first = dataset_name.split('.')[0]
                    # exit(0)
                    depth_dataset = dataset[:].T[:, :, 1]
                    print('depth_dataset.shape',depth_dataset.shape)
                    output_depth = './dset/depth'
                    colored = colorize(depth_dataset)
                    print('colored',colored.shape)
                    # save colored output
                    fpath_colored = os.path.join(output_depth,f'{dataset_name_first}.png')
                    # print('fpath_colored',fpath_colored)
                    img = Image.fromarray(colored)    
                    img.save(fpath_colored)
               
            elif group_name =='seg':
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    print("    数据集名:", dataset_name)
                    print("    数据集形状:", dataset.shape)
                    print("    数据集数据类型:", dataset.dtype)
                    dataset_name_first = dataset_name.split('.')[0]
                    output_depth = './dset/seg'
                    
                    
                    # area = dataset.attrs['area']  # number of pixels in each region
                    label = dataset.attrs['label'] 
                    # print('area',area)
                    print('len(label)',len(label))
                    unique_labels = np.unique(label)
                    num_labels = len(unique_labels)
                    cmap = plt.get_cmap('tab20', num_labels)
                    # 创建一个彩色标签图像，为每个区域赋予唯一的颜色
                    img = Image.open(os.path.join('./flickr30k_img/', dataset_name)).convert('L')
                    
                    print('img',img.size)

                    seg = dataset[:].astype('float32')
                    print('seg.shape',seg.shape)
                    segment_dataset = np.array(Image.fromarray(seg).resize(img.size, Image.NEAREST))
                    # segment_dataset = seg[:].astype('float32')
                    # 创建一个新的图像，用于将分割图的颜色应用到原始图像上
                    colored_image = np.zeros_like(img, dtype=np.uint8)
                    print('colored_image',colored_image.shape)
                    # 调整分割图的尺寸以匹配原始图像的尺寸
                    # segmentation_map = dataset[:img.size[0], :img.size[1]]
                    print('segment_dataset',segment_dataset.shape)

            else:
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    print("    数据集名:", dataset_name)
                    print("    数据集形状:", dataset.shape)
                    print("    数据集数据类型:", dataset.dtype)
                    dataset_name_first = dataset_name.split('.')[0]
                    img = Image.open(os.path.join('./flickr30k_img/', dataset_name)).convert('RGB')
                    print('img.size',img.size)
                    output_depth= './dset/image'
                    fpath_colored = os.path.join(output_depth,f'{dataset_name_first}.jpg')  
                    img.save(fpath_colored) 

#read depth.h5

def read_depth(file_path):
    with h5py.File(file_path, 'r') as f:
        print("HDF5文件信息:")
        print("文件名:", f.filename)
        print("格式版本:", f.libver)
        print("文件模式:", f.mode)
        print(len(f))
        # exit(0)
        # 查看根级组以及组的详细信息
        print("\n根级组:")
        for group_name in f.keys():
            group = f[group_name]
            print("组名:", group_name)
            print("组信息:")
            print("  类型:", type(group))
            print("  ID:", group.id)
            # print("  属性数量:", len(group.attrs))
            print("  数据集数量:", len(group))
            print("  数据集形状:", group.shape)
            print("  数据集数据类型:",group.dtype)
            print(group[:].shape)
            # exit(0)
            # print("  数据集数据类型:",group[1].dtype)
            data_min = np.min(group[0])
            data_max = np.max(group[0])
            data_range = data_max - data_min
            normalized_data = ((group[0] - data_min) / data_range * 255).astype(np.uint8)
            image_path = 'output_image2.png'
            cv2.imwrite(image_path, normalized_data)
            # img0= Image.fromarray(group[0])

            exit(0)
    #read seg.h5
def read_seg(file_path):
    with h5py.File(file_path, 'r') as f:
        print("HDF5文件信息:")
        print("文件名:", f.filename)
        print("格式版本:", f.libver)
        print("文件模式:", f.mode)
        # exit(0)
        # 查看根级组以及组的详细信息
        print("\n根级组:")
        for group_name in f.keys():
            group = f[group_name]
            print("组名:", group_name)
            print("组信息:")
            print("  类型:", type(group))
            print("  ID:", group.id)
            # print("  创建时间:", group.get('created').value)
            # print("  属性数量:", len(group.attrs))
            print("  数据集数量:", len(group))

            # 查看组中的数据集
            # print("  数据集:")
            i =0
            for dataset_name in group.keys():
                
                dataset = group[dataset_name]
                print("    数据集名:", dataset_name)
                print("    数据集形状:", dataset.shape)
                print("    数据集数据类型:", dataset.dtype)
                print("    属性数量:", len(dataset.attrs))
                # print(dataset.attrs['area'])
                # print(dataset.attrs['label'])
                
                # normalized_data = ((dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset)) * 255).astype(np.uint8)
                # # 创建图像
                # img = Image.fromarray(normalized_data)
                # img.save('mask.png')
                i+=1
                if i==5:
                    exit(0)
        # 查看文件级属性
        # print("\n文件级属性:")
        # for attr_name in f.attrs.keys():
        #     print("属性名:", attr_name)
        #     print("属性值:", f.attrs[attr_name])

def main():


    # 打开HDF5文件

    # file_path = './data/dset.h5'  
    # read_dset_h5(file_path)
    
    # file_path = './bg_data_1/seg.h5'
    # read_seg(file_path)

    file_path = './bg_data_1/depth.h5'
    read_depth(file_path)

    # for i in range(0,40):
    #     print(np.random.randn() )
    #     print(int(np.random.randint(0, 3)))
    
    # file_path = './bg_data_1/imnames.cp'  
    # file_path = './renderer_data/models/colors_new.cp'
    # file_path = './renderer_data/models/font_px2pt.cp'  
    # load_cp(file_path)


if __name__ == '__main__':
    main()