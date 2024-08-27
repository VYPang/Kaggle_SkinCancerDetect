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
img_size = conf.dataset.img_size

total_mean = np.zeros(3)
total_std= np.zeros(3)
img_count = 0
for isic_id in tqdm(isic_ids, desc="Calculating mean and std"):
    img = np.array( Image.open(BytesIO(fp_hdf[isic_id][()])) )
    total_mean += np.mean(img, axis=(0, 1))
    total_std += np.std(img, axis=(0, 1))
    img_count += 1

mean = total_mean / img_count
var = total_std / img_count
print(f"Mean: {mean}")
print(f"Std: {np.sqrt(var)}")