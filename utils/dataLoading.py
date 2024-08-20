from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, conf, valid=False, test=False):
        self.df = df
        self.conf = conf
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.img_size = conf.dataset.img_size
        self.mean = conf.dataset.mean
        self.std = conf.dataset.std
        self.valid = valid
        self.test = test
        self.transforms = self.obtain_transforms()
        self.sampler = self.obtain_WeightedRamdomSampler()

    def obtain_WeightedRamdomSampler(self):
        class_count = self.df["target"].value_counts().to_dict()
        total = sum(class_count.values())
        class_weights = {k: total/v for k, v in class_count.items()}
        sample_weights = [class_weights[i] for i in self.targets]
        if self.test:
            return WeightedRandomSampler(sample_weights, self.conf.test.test_sample_size, replacement=False)
        else:
            return WeightedRandomSampler(sample_weights, self.conf.dataset.train_sample_size, replacement=True)

    def obtain_transforms(self):
        if self.valid or self.test:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                        mean=self.mean, 
                        std=self.std, 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
        else:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Downscale(p=0.25),
                A.ShiftScaleRotate(shift_limit=0.1, 
                                scale_limit=0.15, 
                                rotate_limit=60, 
                                p=0.5),
                A.HueSaturationValue(
                        hue_shift_limit=0.2, 
                        sat_shift_limit=0.2, 
                        val_shift_limit=0.2, 
                        p=0.5
                    ),
                A.RandomBrightnessContrast(
                        brightness_limit=(-0.1,0.1), 
                        contrast_limit=(-0.1, 0.1), 
                        p=0.5
                    ),
                A.Normalize(
                        mean=self.mean, 
                        std=self.std, 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
        return transforms
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        target = self.targets[index]
        img = self.transforms(image=img)["image"]

        if self.test:
            return {
                'image': img,
                'target': target,
                'isic_id': isic_id,
            }
        else:
            return {
                'image': img,
                'target': target,
            }

def obtain_dataSet(df, conf, train_hdf_path, test=False):
    if conf.dataset.val_split != 0 and not test:
        val_split = conf.dataset.val_split
        train_sample_size = conf.dataset.train_sample_size
        val_size = int(val_split * train_sample_size)
        target = np.unique(df["target"])
        # calculate the number of samples for each class
        class_count = df["target"].value_counts().to_dict()
        class_weights = {k: round(v/sum(class_count.values()) * val_size) for k, v in class_count.items()}
        val_idx = []
        for t in target:
            df_t = df[df["target"] == t]
            val_idx.extend(df_t.sample(class_weights[t]).index)
        val_df = df.loc[val_idx]
        train_df = df.drop(val_idx)
        train_dataset = ISICDataset(train_df, train_hdf_path, conf, valid=False)
        val_dataset = ISICDataset(val_df, train_hdf_path, conf, valid=True)
    else:
        train_dataset = ISICDataset(df, train_hdf_path, conf, valid=False) if not test else ISICDataset(df, train_hdf_path, conf, test=True)
        val_dataset = None
    return train_dataset, val_dataset

if __name__ == "__main__":
    import pandas as pd
    from omegaconf import OmegaConf
    import cv2

    config_path = "./config/default_config.yaml"
    train_df_path = "./data/train-metadata.csv"
    train_hdf_path = "./data/train-image.hdf5"

    config = OmegaConf.load(config_path)
    train_df = pd.read_csv(train_df_path)
    train_dataset, val_dataset = obtain_dataSet(train_df, config, train_hdf_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=train_dataset.sampler)
    for data in train_loader:
        image = data["image"]
        image = image.squeeze(0).permute(1, 2, 0).numpy() * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("temp.jpg", image)