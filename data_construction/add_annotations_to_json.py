import numpy as np
import json
import copy
from tqdm import tqdm
import os

NEW_DATA_PATH = "./save30_9_test"
OLD_DATA_JSON_PATH = "../GLIGEN_master/dataset/flickr30k.json"

import sys
sys.path.append("..")

categoryid_list=[]

from flickr30k_entities.flickr30k_entities_utils import get_annotations,get_sentence_data

IMG_ID = 0
ANNO_ID = 0
TEXT_ID = 0

class ImageName:
    def __init__(self, img_name):
        self.img_name = img_name
        self.images = []

    def __str__(self):
        return self.img_name

    def __eq__(self, __value: object) -> bool:
        return self.img_name == __value

    def append(self, Image):
        self.images.append(Image)


class Image:
    def __init__(self, img_json):
        self.img_name = img_json["file_name"]
        self.caption = img_json["caption"]
        self.id = img_json["id"]
        self.annotations = []

    def __str__(self):
        return self.img_name

    def __eq__(self, __value: object) -> bool:
        return self.img_name == __value

    def append(self, anno):
        self.annotations.append(anno)

    def get_json(self):
        return {"id": self.id, "file_name": self.img_name, "caption": self.caption}

    def return_filename(self):
        return self.img_name


class Annotation:
    def __init__(self, anno_json):
        self.category_id = anno_json["category_id"]
        self.id = anno_json["id"]
        self.bbox = anno_json["bbox"]
        self.tokens_positive = anno_json["tokens_positive"]
        self.img_id = anno_json["data_id"]

    def __str__(self):
        return str(self.id)

    def get_json(self,file_name,caption):
        category_id=self.get_category_id(file_name,caption)
        categoryid_list.append(category_id)
        return {
            "category_id": category_id,
            "id": self.id,
            "bbox": self.bbox,
            "tokens_positive": self.tokens_positive,
            "data_id": self.img_id,
        }
    
    def get_category_id(self,file_name,caption):
        sentence_files='./flickr30k_entities/Sentences'
        print("file_name",file_name)
        print("caption",caption)
        ori_file_name=file_name.split('.')[0].split('_')[0]
        file_name=ori_file_name+".txt"

        file_path=os.path.join(sentence_files, file_name)
        SentenceData= get_sentence_data(file_path)
        tokens_positive=self.tokens_positive[0]
        print("tokens_positive",tokens_positive)
        phrase=caption[tokens_positive[0]:tokens_positive[1]]
        for senten in SentenceData:
            if senten['sentence']==caption:
                cur_senten=senten
                for one_phrase in cur_senten["phrases"]:
                    if one_phrase['phrase']==phrase:
                        category_id=one_phrase['phrase_id']
                        print("category_id",category_id)
                        return category_id
        
        


class TextAnnotation():
    def __init__(self):
        self.id = None
        self.text = ""
        self.rgb = []
        self.img_id = None
        self.wordBB = None
    def __str__(self):
        return self.id
    def get_json(self):
        return {
            "id":self.id,
            "text":self.text,
            "rgb":self.rgb,
            "wordBB":self.wordBB,
            "data_id":self.img_id,
        }


def process_old_json(old_json):
    old_images_list = old_json["images"]
    old_annotations_list = old_json["annotations"]
    img_name_list = {}
    img_list = {}
    annotation_list = []
    isexistimg = {}
    i =0
    import pdb  
    for image in tqdm(old_images_list,desc = "old_images_list"):
        old_img = Image(image)
        # print('old_img',old_img)
        file_name =  old_img.return_filename()
        if file_name not in isexistimg:
            # print('old_img',file_name)
            # print('id',image["id"])
            
            isexistimg[file_name]= i
            img_list[image["id"]] = old_img
            i += 1

    for anno in tqdm(old_annotations_list,desc = "old_annotations_list"):
        annotation_list.append(Annotation(anno))

    for anno_type in tqdm(annotation_list,desc = "annotation_list"):
        # for i in range(0, len(img_list)):
        #     #  找到了已经有的
        #     if img_list[i].id == anno_type.img_id:
        #         img_list[i].append(anno_type)
        #         break
        # print('anno_type.img_id',anno_type.img_id)
        # pdb.set_trace()
        if anno_type.img_id in img_list:
            img_list[anno_type.img_id].append(anno_type)


    for image_type in tqdm(img_list.values(),desc="img_list"):
        # Imgae 的file name 在不在 img name list里
        # flag = False
        # for i in range(0, len(img_name_list)):
        #     if img_name_list[i] == image_type:
        #         flag = True
        #         img_name_list[i].append(image_type)
        #         break
        # if not flag:
            # img_name_type = ImageName(str(image_type))
            # img_name_type.append(image_type)
            # img_name_list.append(img_name_type)
        if image_type.img_name not in img_name_list:
            img_name_type = ImageName(str(image_type))
            img_name_type.append(image_type)
            img_name_list[image_type.img_name] = img_name_type
        else:
            img_name_list[image_type.img_name].append(image_type)

    # print("[Anno_list:]")
    # for i in annotation_list:
    #     print(i)
    # print("[img_list]:")
    # for i in img_list:
    #     print(i)
    # print("[img_name]:")
    # for i in img_name_list:
    #     print(i)
    # import pdb

    # pdb.set_trace()
    return img_name_list, img_list, annotation_list


def create_new_image_name(new_img_name_list, old_img_name_list):
    global IMG_ID, ANNO_ID
    new_imgname_type_list = []
    # 一个个处理
    # img_name e.g. 17258_1.jpg
    for img_name in tqdm(new_img_name_list):
        new_imgname_type = ImageName(img_name)
        # 获取旧的图像类型
        old_img_name = img_name.split("_")[0] + ".jpg"
        old_img_name_type = old_img_name_list[old_img_name]
        # for i in old_img_name_list:
        #     if i == old_img_name:
        #         old_img_name_type = i
        # 获取新的images，首先加入old_img的images，但是需要更改下名字
        for old_img_type in old_img_name_type.images:
            new_img_type = copy.copy(old_img_type)
            new_img_type.img_name = new_imgname_type.img_name
            # 赋新的ID值
            new_img_type.id = IMG_ID
            IMG_ID += 1
            # 不用原来的anno
            new_img_type.annotations = []
            old_anno_list = old_img_type.annotations
            # 获取旧的anno
            for old_anno in old_anno_list:
                new_anno = copy.copy(old_anno)
                new_anno.id = ANNO_ID
                ANNO_ID += 1
                new_anno.img_id = new_img_type.id
                # 加入更改了id 和IMG id 的annotation
                new_img_type.append(new_anno)
            # 加入新的IMG type ，annotations还没有改变
            new_imgname_type.append(new_img_type)

        # 加入新的imgname type
        new_imgname_type_list.append(new_imgname_type)
    # import pdb

    # pdb.set_trace()
    return new_imgname_type_list


def main():
    # 先读取所有的处理好的图片
    file = open(NEW_DATA_PATH + "/fliter_img_name.txt", "r")
    img_name_in_lines = file.readlines()
    file.close()
    img_name_list = [i.strip() for i in img_name_in_lines]
    print(len(img_name_list))
    with open(OLD_DATA_JSON_PATH) as fr:
        old_json = json.load(fr)
    old_imgname_type_list, old_img_type_list, old_annotation_list = process_old_json(
        old_json
    )

    # 创建一个新的json
    new_json = {}
    # 复制原来的
    new_json["info"] = old_json["info"]
    new_json["licenses"] = old_json["licenses"]
    # 创建新的列表
    new_json["images"] = []
    new_json["text_anno"] = []
    new_json["annotations"] = []

    # 生成新的imgname list
    new_imgname_type_list = create_new_image_name(img_name_list, old_imgname_type_list)
    new_img_list = []
    for img_name_type in tqdm(new_imgname_type_list,desc = "new_imgname_type_list"):
        for img_type in img_name_type.images:
            new_img_list.append(img_type)

    #  把新的anootation放进去
    # 把 text annotation 加入到new_json
    new_text_anno_list = [] # 纯粹方便调试
    for img in tqdm(new_img_list,desc = "new_img_list"):
        new_json["images"].append(img.get_json())
        img_json=img.get_json()
        caption=img_json['caption']
        file_name=img_json['file_name']


        for anno in img.annotations:
            new_json["annotations"].append(anno.get_json(file_name,caption))
        img_name = img.img_name
        # 读取 txt 的 文字区域信息和颜色信息
        text_region_file_contain = open(NEW_DATA_PATH + "/txt/" + img_name + ".txt", "r")

        text_region = eval(text_region_file_contain.read())
        text_region_file_contain.close()
        # fg_rgb_list = np.load(NEW_DATA_PATH + "/fg_color/" + img_name  + ".npy")
        wordBB = np.load(NEW_DATA_PATH + "/wordBB/" + img_name  + ".npy")
        # 对比两个区域的数量相不相同，不相同一定有问题
        try:
            assert len(text_region) == wordBB.shape[2]
        except:
            import pdb 
            pdb.set_trace()
        # 开始加入text_annno
        global TEXT_ID
        

        
        for idx, text in enumerate(text_region):
            # print(len(text_region))
            new_text_anno = TextAnnotation()
            new_text_anno.id = TEXT_ID
            TEXT_ID += 1
            new_text_anno.rgb = [0,0,0]
            new_text_anno.img_id = img.id
            new_text_anno.text = text
            new_text_anno.wordBB = wordBB[:,:,idx].tolist()
            new_text_anno_list.append(new_text_anno)
            new_json["text_anno"].append(new_text_anno.get_json())

    print(len(new_json["text_anno"]))
    with open("json_in_step_6.json", "w") as fr:
        json.dump(new_json, fr,indent = 4)
    print(list(set(categoryid_list)))
    # print("len(print(list(set(categoryid_list))))",len(list(set(categoryid_list))))
    # print('finish write!!')

if __name__ == "__main__":
    main()
