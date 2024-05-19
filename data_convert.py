####################################################################
# Convert Original fundus data (.npz) to images and labels (.png)
####################################################################

import os
import numpy as np
import cv2
from tqdm import tqdm

# path to Harvard-FairSeg dataset
datadir = '/DATA_EDS/datasets/MedicalImage/data/' 
targetdir = '/DATA_EDSdatasets/MedicalImage/'

files = [f for f in os.listdir(datadir) if f.endswith('.npz')]

for file_name in tqdm(files, desc="Processing files", unit="file"):
    file_path = os.path.join(datadir, file_name)
    
    data = np.load(file_path)

    num = file_name.split('_')[1].split('.')[0]
    gray_image, label = data['fundus'], data['disc_cup_borders']

    height, width = gray_image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = gray_image  # Red channel
    rgb_image[:, :, 1] = gray_image  # Green channel
    rgb_image[:, :, 2] = gray_image  # Blue channel

    cv2.imwrite(targetdir+'/images/'+num+'.png', rgb_image)

    # map rule
    # cmap = {0: [255, 255, 255], -1: [0, 0, 255], -2: [255, 0, 0]}
    cmap = {0: [0,0,0], -1: [1, 1, 1], -2: [2, 2, 2]}
    height, width = label.shape
    rgb_label = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            rgb_label[i, j] = cmap[label[i, j]]

    cv2.imwrite(targetdir+'/labels/'+num+'.png', label)
    


