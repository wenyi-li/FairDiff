import os,sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from data.dataset import *
from utils.misc import *
from data.data import *
from models.vae_flow_genz import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from pc2label import *

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow')
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=6)
parser.add_argument('--sample_num_points', type=int, default=512)
parser.add_argument('--kl_weight', type=float, default=0.0001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--resume_path', type=str, default='',
                    help = 'model ckpt path')
parser.add_argument('--generate_num', type=int, default=100)
parser.add_argument('--dataset_path', type=str, default='./data/pointcloud')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--categories', type=str, default='gender',
                    choices = ['ethnicity', 'gender', 'language', 'maritalstatus',
                    'race'])
parser.add_argument('--attributes', type=str, default='Female',
                    choices = ['Female', 'Male'])
parser.add_argument('--scale_mode', type=str, default='global_unit',
                    choices = ['shape_unit', 'global_unit', 'no'])
parser.add_argument('--train_batch_size', type=int, default=48)
parser.add_argument('--val_batch_size', type=int, default=48)

parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()
seed_all(args.seed)

# Model
if args.model == 'flow':
    model = FlowVAE_Genz(args).to(args.device)
if args.spectral_norm:
    add_spectral_norm(model)

# Resume
resume_path = args.resume_path
ckpt_resume = torch.load(resume_path)
model.load_state_dict(ckpt_resume['state_dict'])

generate_num = args.generate_num
itera_num = math.ceil( generate_num / args.val_batch_size)
remain_last =  generate_num - (itera_num - 1) * args.val_batch_size

def generate_filenames(start = 1, end = generate_num, batch_size = args.val_batch_size):
    filenames = []
    for i in range(start, end + 1):
        filename = f"{i:06}.xyz"
        filenames.append(filename)
        if i % batch_size == 0 or i == end:
            yield filenames
            filenames = []
            
#1. noise generate point cloud
batch_filenames = list(generate_filenames())  # Create a list of all batches
for i in range(itera_num):
    z = torch.randn([args.train_batch_size, args.latent_dim]).to(args.device)
    y = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
    stats_dir = os.path.join(args.dataset_path, args.categories, args.attributes + '_stats')
    stats_save_path = os.path.join(stats_dir, 'stats_' + args.attributes + '.pt')
    stats = (torch.load(stats_save_path))
    shift = stats['mean'].reshape(1, 3)
    scale = stats['std'].reshape(1, 1)
    y = y.cpu()
    x = y * scale + shift

    data_dir = os.path.join(args.result_path, args.categories, args.attributes, "noise2pc")
    os.makedirs(data_dir, exist_ok=True)

    name = batch_filenames[i]
    aaa = 1
    if i == itera_num - 1:
        for j in range(remain_last):
            filename = name[j]
            filename = os.path.join(data_dir, filename)
            np_array = x[j].squeeze(0).numpy()
            np.savetxt(filename, np_array, fmt='%f', delimiter=' ')
    else:
        for j in range(args.train_batch_size):
            filename = name[j]
            filename = os.path.join(data_dir, filename)
            np_array = x[j].squeeze(0).numpy()
            np.savetxt(filename, np_array, fmt='%f', delimiter=' ')
            
#2.point cloud generate label
savedir = os.path.join(args.result_path, args.categories, args.attributes, "label")
os.makedirs(savedir, exist_ok=True)
files = [f for f in os.listdir(data_dir) if f.endswith('.xyz')]
for file_name in tqdm(files, desc="Processing files", unit="file"):
    file_path = os.path.join(data_dir, file_name)
    points = read_point_cloud(file_path)
    # 1. Convert point cloud to image_blue and image_redW
    image_blue, image_red = point_cloud_to_image(points, width=798, height=664)
    # 2. Corrosion expands to fill the blue and red areas
    mask_blue = blue_closed_operations(image_blue, kernel_size=15, iterations=20)
    mask_red = red_closed_operations(image_red, kernel_size=15, iterations=20)
    # 3. Combine blue and red areas
    combined_img = combine(mask_blue, mask_red)
    basename = file_name.split('.')[0] + '.png'
    output_path = os.path.join(savedir, basename)
    cv2.imwrite(output_path, combined_img)






