import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import math
import cv2
import torch
import lpips
import csv
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.nn.functional import cosine_similarity
from utils.misc import *
from metrics.evaluation_image import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=128)
# Sampling
parser.add_argument('--seed', type=int, default=9988)
parser.add_argument('--ref_folder', type=str, help='original image path')
parser.add_argument('--gen_folder', type=str, help='generate image path')
parser.add_argument('--csv_path', type=str, default='result.csv', help='save path')

args = parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, image_paths, resolution=512):
        self.image_paths = sorted([os.path.join(image_paths, f) for f in os.listdir(image_paths)])
        self.resolution = resolution
       
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        loaded_image = lpips.load_image(img_path) 
        loaded_image = cv2.resize(loaded_image, (self.resolution, self.resolution))
        img = lpips.im2tensor(loaded_image)
        return img, img_path

model = resnet50(pretrained=True).to(args.device)
model.eval()
ref_folder = args.ref_folder
gen_folder = args.gen_folder
ref_dataset = ImageDataset(ref_folder)
gen_dataset = ImageDataset(gen_folder)

ref_dataloader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
gen_dataloader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print("calculate all_ref_feature...")
all_ref_feature = []
for batch_img, batch_path in tqdm(ref_dataloader, desc="Extracting reference image features"):
    with torch.no_grad():
        batch_img = batch_img.to('cuda')
        batch_img = batch_img.squeeze(1).contiguous()
        features = model(batch_img)
        all_ref_feature.append(features.detach().cpu())
all_ref_feature = torch.cat(all_ref_feature, dim=0)

print("calculate all_gen_feature...")
all_gen_feature = []
for batch_img, batch_path in tqdm(gen_dataloader, desc="Extracting generation image features"):
    with torch.no_grad():
        batch_img = batch_img.to('cuda')
        batch_img = batch_img.squeeze(1).contiguous()
        features = model(batch_img)
        all_gen_feature.append(features.detach().cpu())
all_gen_feature = torch.cat(all_gen_feature, dim=0)

# Compute metrics
results = compute_all_metrics(all_gen_feature.to(args.device), all_ref_feature.to(args.device), args.batch_size)
results = {k:v.item() for k, v in results.items()}
with open(args.csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for key, value in results.items():
        writer.writerow([key, value])