import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import pandas as pd
from omegaconf import OmegaConf
import cv2

hdf_path = "./data/train-image.hdf5"
df_path = "./data/train-metadata.csv"
conf_path = "./config/default_config.yaml"

fp_hdf = h5py.File(hdf_path, mode="r")
conf = OmegaConf.load(conf_path)
df = pd.read_csv(df_path)
isic_ids = df["isic_id"].values
img_size = conf.img_size

img_rec = None
for isic_id in tqdm(isic_ids, desc="Calculating mean and std"):
    img = np.array( Image.open(BytesIO(fp_hdf[isic_id][()])) )
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = img.reshape((1, img_size, img_size, 3))
    if img_rec is None:
        img_rec = img
    else:
        img_rec = np.concatenate((img_rec, img), axis=0)
mean = np.mean(img_rec, axis=(0, 1, 2))
std = np.std(img_rec, axis=(0, 1, 2))
print(f"Mean: {mean}")
print(f"Std: {std}")