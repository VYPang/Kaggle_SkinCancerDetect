from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py

class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, conf, valid=False):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.img_size = conf.img_size
        self.valid = valid
        self.obtain_transforms()
    
    def obtain_transforms(self):
        if self.valid:
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
        else:
            self.transforms = A.Compose([
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
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        target = self.targets[index]
        img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target,
        }

if __name__ == "__main__":
    import pandas as pd
    from omegaconf import OmegaConf
    import cv2

    config_path = "./config/default_config.yaml"
    train_df_path = "./data/train-metadata.csv"
    train_hdf_path = "./data/train-image.hdf5"

    config = OmegaConf.load(config_path)
    train_df = pd.read_csv(train_df_path)
    train_dataset = ISICDataset(train_df, train_hdf_path, config, valid=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True)
    for data in train_loader:
        image = data["image"]
        # save image
        image = image.squeeze(0).permute(1, 2, 0).numpy() * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("temp.jpg", image)