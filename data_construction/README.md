# LTOS
Layout-controllable Text-Object Synthesis via Adaptive Cross-attention Fusions

## Requirements
get_depth: ./get_depth/environment.yml
get_segmentation: matlab
render_text: ./requirements.txt

## LTOS dataset 
[link](https://baidu.com)


## Data process
### 1.Download Flickr30K dataset
Flickr Image dataset &nbsp;&nbsp; [Download](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/ "Download form kaggle")

### 2.Process Flickr30K dataset
The origin dataset file is in TSV format and we should process it.

Run `tsv_to_json.py` to transform tsv file to json file (drop images).

Run `tsv_to_image.py` to extract images from tsv file.
### 3.Get the depth and segmentation maps of the Flickr30k
Run  `./get_depth/get_depth_h5.py` to extract the depth maps
Run  `./get_segmentation/run_ucm.m ` and `./get_segmentation/floodFill.py`  to extract the segmentation maps

### 4.Render text in the images
After extracting images, we then run `gen_into_dir.py` to add text to images of Flickr30K extracted in step 2.
```bash
python gen_into_dir.py  
```
Some hyper-parameters are defined in files.

### 5.Filter out low-quality images
Because the added images may have drawbacks such as being blurry, having small fonts, or being too similar to the environment, we filter the images generated in step 3.
```bash
python img_filter.py
```
### 6.Add text and object_catagory annotations to json
The original dataset includes caption information for images and object location information. Now, we are adding text information to the JSON file generated in step 2.

The text information has already been generated in step 3. We simply need to add it to the JSON file.
```bash
python add_annotations_to_json.py 
```

### 7.Generate grounding information
Before generating the new TSV file, we need to process the grounding information first.
Images generated in step 4 are processed by CLIP model to get embedded tensors.

```bash
python process_grounding.py --json_path json_in_step_6.json --image_root images_in_step_3 --folder folder_in_step_3
```
### 8.Integrate to form a new dataset
Now, we integrate all the information to form our new dataset, namely the LTOS dataset.

```bash
python mydata_to_tsv.py --image_root images_in_step_3 --json_path json_in_step_6.json --tsv_path new_tsv_save_path.tsv --mask_root mask_in_file_of_step_3 --annotation_embedding_path file_in_step_7
```

