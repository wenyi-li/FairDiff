import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

def process_image(n):
    file = f"{n:06d}.png"
    img_path = os.path.join(dir, attr_name, 'image', file)
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.resize(gray_image, (798, 664))

    label_path = os.path.join(dir, attr_name, 'label', file)
    label = cv2.imread(label_path)
    resized_label = cv2.resize(label, (798, 664), interpolation=cv2.INTER_NEAREST)
    pixel_map = {
        (255, 255, 255): 0.0,
        (0, 0, 255): -1.0,
        (255, 0, 0): -2.0
    }
    new_label = np.zeros_like(resized_label[:, :, 0], dtype=np.float64)
    for pixel_value, label_value in pixel_map.items():
        mask = np.all(resized_label == np.array(pixel_value), axis=-1)
        new_label[mask] = label_value

    fundus = new_image
    disc_cup_borders = new_label

    num = f"{n:06d}.npz"
    save_path = os.path.join(dir, attr_name, 'data', num)
    np.savez(save_path,
             fundus=fundus,
             disc_cup_borders=disc_cup_borders,
             race=race_attr_map[attr_name])

dir = 'path/to/data'
race_attr_map = {'Asian':2,'Black':3,'White':7}
attr_name = 'Asian'

num_processes = os.cpu_count()
max_value = 50

with Pool(processes=num_processes) as pool:
    list(tqdm(pool.imap(process_image, range(max_value)), total=max_value))
