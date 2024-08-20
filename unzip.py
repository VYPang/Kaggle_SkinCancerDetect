import zipfile
from tqdm import tqdm

# Unzip the file with progress bar
with zipfile.ZipFile('isic-2024-challenge.zip', 'r') as z:
    # Get the total number of files in the zip archive
    total_files = len(z.namelist())
    
    # Extract files with progress bar
    with tqdm(total=total_files, unit='file', ncols=80, desc='Extracting') as pbar:
        for filename in z.namelist():
            z.extract(filename, path=r"./data")
            pbar.update(1)