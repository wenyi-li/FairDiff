import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_fairseg import FairSeg_dataset, TestGenerator
from utils import test_single_volume, test_single_image, equity_scaled_perf, equity_scaled_std_perf
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from inference import inference

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str,
                    default='/path/to/your/datasets/MedicalImage/data', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='FairSeg', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='lists/FairSeg_final', help='list dir')
parser.add_argument('--attribute', type=str,
                    default='race', help='attribute labels')
parser.add_argument('--center_crop_size', type=int, default=512,
                    help='center croped image size | 512 for slo, 420 for oct fundus')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224,
                    help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",
                    help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=0,
                    help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str,
                    default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--lora_ckpt', type=str, default='',
                    help='The checkpoint from LoRA')
parser.add_argument('--exp_name', type=str, default='', help='')
parser.add_argument('--output', type=str, default='', help='')
parser.add_argument('--epoch', type=str, default='', help='')
args = parser.parse_args()


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'FairSeg': {
            'Dataset': FairSeg_dataset,
            'data_dir': args.datadir,
            'num_classes': args.num_classes,
        },
    }
    dataset_name = args.dataset

    args.is_pretrain = True

    args.exp = 'TU_' + dataset_name + str(args.img_size)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(
            args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes).cuda()

    checkpoint = torch.load(args.lora_ckpt)

    new_state_dict = {k.replace('module.', ''): v for k,
                      v in checkpoint.items()}

    net.load_state_dict(new_state_dict)

    log_folder = args.output + '/' + args.exp_name + '/' + args.epoch

    os.makedirs(log_folder, exist_ok=True)

    log_file_path = log_folder + '/'+"log.txt"

    logging.basicConfig(level=logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(str(args))

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(
            args.test_save_dir, args.exp, args.exp_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    if args.attribute == 'race' or args.attribute == 'language':
        no_of_attr = 3
    else:
        no_of_attr = 2

    inference(
        args, net, dataset_config[dataset_name], test_save_path, no_of_attr)
