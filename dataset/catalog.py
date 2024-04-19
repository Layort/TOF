import os 

class DatasetCatalog:
    def __init__(self, ROOT):


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.FlickrGrounding = {
            "target": "dataset.tsv_dataset.TSVDatasetNew",
            "train_params":dict(
                tsv_path=os.path.join('/TOF/DATA','LTOS_train.tsv'),
            ),
        }

        self.FlickrGrounding_test = {
            "target": "dataset.tsv_dataset.TSVDatasetNew",
            "train_params":dict(
                tsv_path=os.path.join('/TOF/DATA','LTOS_test.tsv'),
            ),
        }





