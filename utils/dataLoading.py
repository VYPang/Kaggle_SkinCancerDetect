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
import cupy as cp

class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, conf, valid=False, test=False):
        self.df = df
        self.conf = conf
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.test = test
        self.img_size = conf.dataset.img_size
        self.mean = conf.dataset.mean
        self.std = conf.dataset.std
        self.valid = valid
        self.transforms = self.obtain_transforms()
        if not test:
            self.targets = df['target'].values
            self.sampler = self.obtain_WeightedRamdomSampler()

    def obtain_WeightedRamdomSampler(self):
        class_count = self.df["target"].value_counts().to_dict()
        total = sum(class_count.values())
        class_weights = {k: total/v for k, v in class_count.items()}
        sample_weights = [class_weights[i] for i in self.targets]
        return WeightedRandomSampler(sample_weights, self.conf.dataset.train_sample_size, replacement=True)

    def obtain_transforms(self):
        if self.valid:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(),
                ToTensorV2()], p=1.)
        else:
            transforms = A.Compose([
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, p=0.5),
                A.ColorJitter(contrast=0.2, p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4),
                    A.ElasticTransform(alpha=1, sigma=50),
                ], p=0.7),

                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(self.img_size, self.img_size),
                A.CoarseDropout(max_height=int(self.img_size * 0.1), max_width=int(self.img_size * 0.1), max_holes=1, p=0.7),
                A.Normalize(),
                ToTensorV2()], p=1.)
        return transforms
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        img = self.transforms(image=img)["image"]
        if not self.test:
            target = self.targets[index]
            return {
                'image': img,
                'target': target,
                'isic_id': isic_id,
            }
        else:
            return {
                'image': img,
                'isic_id': isic_id,
            }

# When test = True, conf.dataset.val_split must be 0
def obtain_dataSet(df, conf, train_hdf_path, test=False):
    if conf.dataset.val_split != 0:
        val_split = conf.dataset.val_split
        train_sample_size = conf.dataset.train_sample_size
        val_size = int(val_split * train_sample_size)
        target = np.unique(df["target"])
        # calculate the number of samples for each class
        class_count = df["target"].value_counts().to_dict()
        class_weights = {k: round(v/sum(class_count.values()) * val_size) for k, v in class_count.items()}
        # ensure at leat 2
        for k, v in class_weights.items():
            if v < 5:
                class_weights[k] = 5
        val_idx = []
        for t in target:
            df_t = df[df["target"] == t]
            val_idx.extend(df_t.sample(class_weights[t]).index)
        val_df = df.loc[val_idx]
        train_df = df.drop(val_idx)
        train_dataset = ISICDataset(train_df, train_hdf_path, conf, valid=False)
        val_dataset = ISICDataset(val_df, train_hdf_path, conf, valid=True)
    else:
        train_dataset = ISICDataset(df, train_hdf_path, conf, valid=False, test=test)
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