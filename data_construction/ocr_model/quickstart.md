---


---
## 模型推理
- 使用图像的url，或准备图像文件。
- 输入下列代码。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

### 使用url
img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
result = ocr_recognition(img_url)
print(result)

### 使用图像文件
### 请准备好名为'ocr_recognition.jpg'的图像文件
# img_path = 'ocr_recognition.jpg'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)
# print(result)
```
更多关于模型加载和推理的问题参考[模型的推理Pipeline](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline)。

## 模型训练
### 模型微调/训练示例
#### 训练数据集准备
示例采用[ICDAR13手写数据集](https://modelscope.cn/datasets/damo/ICDAR13_HCTR_Dataset/summary)，已制作成lmdb，数据格式如下
```
'num-samples': number,
'image-000000001': imagedata,
'label-000000001': string,
...
```
详情可下载解析了解。

#### 配置训练参数并进行微调/训练
参考代码及详细说明如下
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile, DownloadMode

### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中 

model_id = 'damo/cv_convnextTiny_ocr-recognition-general_damo' 
cache_path = snapshot_download(model_id) # 模型下载保存目录
config_path = os.path.join(cache_path, ModelFile.CONFIGURATION) # 模型参数配置文件，支持自定义
cfg = Config.from_file(config_path)

# 构建数据集，支持自定义
train_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset', 
    split='test',
    namespace='damo',
    test_mode=False)

train_dataset = MsDataset.load( 
    dataset_name=train_data_cfg.name,
    split=train_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

test_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset',
    split='test',
    namespace='damo',
    test_mode=True)

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

tmp_dir = tempfile.TemporaryDirectory().name # 模型文件和log保存位置，默认为"work_dir/"

# 自定义参数，例如这里将max_epochs设置为15，所有参数请参考configuration.json
def _cfg_modify_fn(cfg):
    cfg.train.max_epochs = 15
    return cfg

####################################################################################

'''
使用本地文件
    lmdb: 
        构建包含下列信息的lmdb文件 (key: value)
        'num-samples': 总样本数,
        'image-000000001': 图像的二进制编码,
        'label-000000001': 标签序列的二进制编码,
        ...
        image和label后的index为9位并从1开始
下面为示例 (local_lmdb为本地的lmdb文件)
'''

# train_dataset = MsDataset.load( 
#     dataset_name=train_data_cfg.name,
#     split=train_data_cfg.split,
#     namespace=train_data_cfg.namespace,
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
#     local_lmdb='./local_lmdb')

# test_dataset = MsDataset.load(
#     dataset_name=test_data_cfg.name,
#     split=test_data_cfg.split,
#     namespace=train_data_cfg.namespace,
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
#     local_lmdb='./local_lmdb')

####################################################################################

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    work_dir=tmp_dir,
    cfg_modify_fn=_cfg_modify_fn)

# 模型训练
trainer = build_trainer(name=Trainers.ocr_recognition, default_args=kwargs)
trainer.train()
```

#### 用训练/微调后的模型进行识别
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

ep_num = 3  # 选择模型checkpoint
cmd = 'cp {} {}'.format('./work_dir/epoch_%d.pth' % ep_num, './work_dir/output/pytorch_model.pt')  # 'work_dir'为configuration中设置的路径，'output'为输出默认路径
os.system(cmd)
ocr_recognition = pipeline(Tasks.ocr_recognition, model='./work_dir/output' )
result = ocr_recognition('http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_icdar13.jpg')
print(result)
```
更多使用说明请参阅[ModelScope文档中心](http://www.modelscope.cn/#/docs)。

---


---
## 下载并安装ModelScope library
更多关于下载安装ModelScope library的问题参考[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。

```python
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
