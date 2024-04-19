import numpy as np
import json
import copy
from tqdm import tqdm
from tsv import TSVFile

NEW_DATA_PATH = "./new_generate_data"
OLD_DATA_JSON_PATH = "old.json"
OLD_TSV_PATH = "old.tsv"


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
        self.id = img_json["data_id"]
        self.annotations = []

    def __str__(self):
        return self.img_name

    def __eq__(self, __value: object) -> bool:
        return self.img_name == __value

    def append(self, anno):
        self.annotations.append(anno)

    def get_json(self):
        return {"id": self.id, "file_name": self.img_name, "caption": self.caption}


class Annotation:
    def __init__(self, anno_json):
        self.category_id = anno_json["category_id"]
        self.id = anno_json["id"]
        self.bbox = anno_json["bbox"]
        self.tokens_positive = anno_json["tokens_positive"]
        self.img_id = anno_json["data_id"]

    def __str__(self):
        return str(self.id)

    def get_json(self):
        return {
            "category_id": self.category_id,
            "id": self.id,
            "bbox": self.bbox,
            "tokens_positive": self.tokens_positive,
            "data_id": self.img_id,
        }

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
    for image in tqdm(old_images_list,desc = "old_images_list"):
        old_img = Image(image)
        img_list[image["data_id"]] = old_img

    for anno in tqdm(old_annotations_list,desc = "old_annotations_list"):
        annotation_list.append(Annotation(anno))

    for anno_type in tqdm(annotation_list,desc = "annotation_list"):
        img_list[anno_type.img_id].append(anno_type)


    for image_type in tqdm(img_list.values(),desc="img_list"):
        # Imgae 's file name not in img name list
        if image_type.img_name not in img_name_list:
            img_name_type = ImageName(str(image_type))
            img_name_type.append(image_type)
            img_name_list[image_type.img_name] = img_name_type
        else:
            img_name_list[image_type.img_name].append(image_type)

    return img_name_list, img_list, annotation_list


def tsv_to_json(tsvfile):
    old_json = {"images":[], "annotations":[]}
    for i in tqdm(range(len(tsvfile)),desc="read tsv"):
        item = eval(tsvfile[i][1])
        old_json["images"].append({
            'data_id':item['data_id'], 
            'image':item['data_id'], 
            'file_name':item['file_name'], 
            'caption': item['caption']
        })
        for anno in item["annos"]:
            old_json["annotations"].append(anno)
    return old_json

def create_new_image_name(new_img_name_list, old_img_name_list):
    global IMG_ID, ANNO_ID
    new_imgname_type_list = []
    # process one by one
    # img_name e.g. 17258_1.jpg
    for img_name in tqdm(new_img_name_list):
        new_imgname_type = ImageName(img_name)
        # get old image 
        old_img_name = img_name.split("_")[0] + ".jpg"
        old_img_name_type = old_img_name_list[old_img_name]
        # generate new
        for old_img_type in old_img_name_type.images:
            new_img_type = copy.copy(old_img_type)
            new_img_type.img_name = new_imgname_type.img_name
            # assign new id
            new_img_type.id = IMG_ID
            IMG_ID += 1
            new_img_type.annotations = []
            old_anno_list = old_img_type.annotations
            # get old annotation
            for old_anno in old_anno_list:
                new_anno = copy.copy(old_anno)
                new_anno.id = ANNO_ID
                ANNO_ID += 1
                new_anno.img_id = new_img_type.id
                # new annotation
                new_img_type.append(new_anno)
            new_imgname_type.append(new_img_type)
        # add new imgname type
        new_imgname_type_list.append(new_imgname_type) 
    return new_imgname_type_list


def main():
    # read all preprocess images
    file = open(NEW_DATA_PATH + "/img_name.txt", "r")
    img_name_in_lines = file.readlines()
    file.close()
    img_name_list = [i.strip() for i in img_name_in_lines]
    print(len(img_name_list))
    
    tsvfile = TSVFile(OLD_TSV_PATH)
    old_json = tsv_to_json(tsvfile)
    
    old_imgname_type_list, old_img_type_list, old_annotation_list = process_old_json(
        old_json
    )

    new_json = {}
    new_json["info"] = []
    new_json["licenses"] = []
    new_json["images"] = []
    new_json["text_anno"] = []
    new_json["annotations"] = []

    # generate imgname list
    new_imgname_type_list = create_new_image_name(img_name_list, old_imgname_type_list)
    new_img_list = []
    for img_name_type in tqdm(new_imgname_type_list,desc = "new_imgname_type_list"):
        for img_type in img_name_type.images:
            new_img_list.append(img_type)

    #  put in new annotations
    # put text annotation to new_json
    new_text_anno_list = [] # 纯粹方便调试
    for img in tqdm(new_img_list,desc = "new_img_list"):
        try:
            img_name = img.img_name
            # read text region and color
            text_region_file_contain = open(NEW_DATA_PATH + "/txt/" + img_name + ".txt", "r")
            text_region = eval(text_region_file_contain.read())
            text_region_file_contain.close()
            # fg_rgb_list = np.load(NEW_DATA_PATH + "/fg_color/" + img_name  + ".npy")
            wordBB = np.load(NEW_DATA_PATH + "/wordBB/" + img_name  + ".npy")
            assert len(text_region) == wordBB.shape[2]
            new_json["images"].append(img.get_json())
            for anno in img.annotations:
                new_json["annotations"].append(anno.get_json())
        except:
            continue
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
    with open("new.json", "w") as fr:
        json.dump(new_json, fr,indent = 4)


if __name__ == "__main__":
    main()
