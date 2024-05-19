import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume, test_single_image
from importlib import import_module
from segment_anything import sam_model_registry
from icecream import ic
from torchvision import transforms
from utils import equity_scaled_perf,equity_scaled_std_perf
from datasets.dataset_fairseg_abs import FairSeg_dataset, TestGenerator 



import pandas as pd
import logging
import numpy as np
import csv

import datetime


class_to_name = {1: 'cup', 2: 'disc'}


def inference(args, multimask_output, db_config, model, test_save_path=None, no_of_attr=3):
    db_test = db_config['Dataset'](base_dir=args.datadir, args=args, split='test', attr_label=args.attribute, \
                                    transform=transforms.Compose([TestGenerator(output_size=[args.img_size, args.img_size], \
                                                                             low_res=[224, 224], center_crop_size=args.center_crop_size, use_normalize=True)]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1) 
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    
    cdr_overall = 0.0
    auc_overall = 0.0

    cdr_by_attr = [ [] for _ in range(no_of_attr) ]
    
    auc_by_attr = [ [] for _ in range(no_of_attr) ]

    dice_by_attr = [ [] for _ in range(no_of_attr) ]
    hd_by_attr = [ [] for _ in range(no_of_attr) ]
    jc_by_attr = [ [] for _ in range(no_of_attr) ]
    dice_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    dice_by_attr_disc = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_disc = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_disc = [ [] for _ in range(no_of_attr) ]
    dice_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_rim = [ [] for _ in range(no_of_attr) ]

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
      
        # h, w = sampled_batch['image'].shape[2:]
        # print(h)
        # print(w)
        image, label, case_name, attr_label = sampled_batch['image'], sampled_batch['label'], \
            sampled_batch['pid'], sampled_batch['attr_label']
        
        metric_i, cdr_dist, auc_score = test_single_image(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, \
                                        attr_label=attr_label, idx=i_batch)
        
        
        metric_list += np.array(metric_i)
        auc_overall += auc_score
        cdr_overall += cdr_dist

        attr_label = attr_label.detach().cpu().numpy().item()

        if attr_label != -1:
            dice_by_attr[attr_label].append(np.mean(metric_i, axis=0)[0])
            hd_by_attr[attr_label].append(np.mean(metric_i, axis=0)[1])
            jc_by_attr[attr_label].append(np.mean(metric_i, axis=0)[2])
            cdr_by_attr[attr_label].append(cdr_dist)
            auc_by_attr[attr_label].append(auc_score)

            # compute for rim 
            dice_by_attr_rim[attr_label].append(metric_i[0][0])
            hd_by_attr_rim[attr_label].append(metric_i[0][1])
            jc_by_attr_rim[attr_label].append(metric_i[0][2])

            # compute for cup 
            dice_by_attr_cup[attr_label].append(metric_i[1][0])
            hd_by_attr_cup[attr_label].append(metric_i[1][1])
            jc_by_attr_cup[attr_label].append(metric_i[1][2])
            
            # # compute for cup 
            # dice_by_attr_disc[attr_label].append(metric_i[1][0])
            # print(metric_i[1][0])
            # hd_by_attr_disc[attr_label].append(metric_i[1][1])
            # jc_by_attr_disc[attr_label].append(metric_i[1][2])
            # # compute for rim 
            # dice_by_attr_rim[attr_label].append(metric_i[2][0])
            # print(metric_i[2][0])
            # hd_by_attr_rim[attr_label].append(metric_i[2][1])
            # jc_by_attr_rim[attr_label].append(metric_i[2][2])
            # exit()
    
    
    metric_list = metric_list / len(db_test)
    mean_auc = auc_overall / len(db_test)
    mean_cdr_dist = cdr_overall / len(db_test) 
   
    # print(metric_list.shape)
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    cup_overall_dice = metric_list[1][0]
    rim_overall_dice = metric_list[0][0]

    cup_overall_hd95 = metric_list[1][1]
    rim_overall_hd95 = metric_list[0][1]

    cup_overall_jaccard = metric_list[1][2]
    rim_overall_jaccard = metric_list[0][2]

    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    experiment_data = {
    'Experiment Name': args.exp_name,
    'Epoth' : args.epoch,
    'Time': current_time_str, 
    'Attribute': args.attribute,
    'Cup Es-Dice':0.0,
    'Cup Overall Dice':0.0,
    'Cup Es-IoU':0.0,
    'Cup Overall IoU':0.0,
    '0-attr dice for Cup':0.0,
    '1-attr dice for Cup':0.0,
    '2-attr dice for Cup':0.0,
    '0-attr IoU for Cup':0.0,
    '1-attr IoU for Cup':0.0,
    '2-attr IoU for Cup':0.0,
    'Rim Es-Dice':0.0,
    'Rim Overall Dice':0.0,
    'Rim Es-IoU':0.0,
    'Rim Overall IoU':0.0,
    '0-attr dice for Rim':0.0,
    '1-attr dice for Rim':0.0,
    '2-attr dice for Rim':0.0,
    '0-attr IoU for Rim':0.0,
    '1-attr IoU for Rim':0.0,
    '2-attr IoU for Rim':0.0,
    }
    
    logging.info('--------- Overall Performance for Attribute: {} -----------'.format(args.attribute))

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr[one_attr]
        one_attr_hd_list = hd_by_attr[one_attr]
        one_attr_jc_list = jc_by_attr[one_attr]
        one_attr_auc_list = auc_by_attr[one_attr]
        one_attr_cdr_list = cdr_by_attr[one_attr]

        logging.info(f'{one_attr}-attr overall dice: {np.mean(one_attr_dice_list):.4f}')
        logging.info(f'{one_attr}-attr overall hd95: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr overall Jaccard/IoU: {np.mean(one_attr_jc_list):.4f}')
        # logging.info(f'{one_attr}-attr overall AUC: {np.mean(one_attr_auc_list):.4f}')
        # logging.info(f'{one_attr}-attr overall CDR Distance: {np.mean(one_attr_cdr_list):.4f}')



    logging.info('--------- Cup Performance for Attribute: {} -----------'.format(args.attribute))
    
    logging.info(f'Cup Overall Dice: {cup_overall_dice:.4f}')
    experiment_data['Cup Overall Dice'] = cup_overall_dice
    logging.info(f'Cup Overall hd95: {cup_overall_hd95:.4f}')
    logging.info(f'Cup Overall IoU: {cup_overall_jaccard:.4f}')
    experiment_data['Cup Overall IoU'] = cup_overall_jaccard
    
    es_cup_dice = equity_scaled_perf(dice_by_attr_cup, cup_overall_dice, no_of_attr)
    es_std_cup_dice = equity_scaled_std_perf(dice_by_attr_cup, cup_overall_dice, no_of_attr)
    es_cup_iou = equity_scaled_perf(jc_by_attr_cup, cup_overall_jaccard,no_of_attr)
    es_std_cup_iou= equity_scaled_std_perf(jc_by_attr_cup, cup_overall_jaccard,no_of_attr)
    
    logging.info(f'Cup Es-Dice: {es_cup_dice:.4f}')
    
    logging.info(f'Cup Es-IoU: {es_cup_iou:.4f}')
    
    
    
    logging.info(f'Cup Es-Std-Dice: {es_std_cup_dice:.4f}')
    experiment_data['Cup Es-Dice'] = es_std_cup_dice
    logging.info(f'Cup Es-Std-IoU: {es_std_cup_iou:.4f}')
    experiment_data['Cup Es-IoU'] = es_std_cup_iou

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_cup[one_attr]
        one_attr_hd_list = hd_by_attr_cup[one_attr]
        one_attr_jc_list = jc_by_attr_cup[one_attr]
        
        logging.info(f'{one_attr}-attr dice for Cup: {np.mean(one_attr_dice_list):.4f}')
        experiment_data[f'{one_attr}-attr dice for Cup'] = np.mean(one_attr_dice_list)
        logging.info(f'{one_attr}-attr hd95 for Cup: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr Jaccard/IoU for Cup: {np.mean(one_attr_jc_list):.4f}')
        experiment_data[f'{one_attr}-attr IoU for Cup'] = np.mean(one_attr_jc_list)

    # logging.info('--------- Disc Performance for Attribute: {} -----------'.format(args.attribute))
    
    # for one_attr in range(no_of_attr):
    #     one_attr_dice_list = dice_by_attr_cup[one_attr]
    #     one_attr_hd_list = hd_by_attr_cup[one_attr]
    #     one_attr_jc_list = jc_by_attr_cup[one_attr]

    #     logging.info(f'{one_attr}-attr dice for Disc: {np.mean(one_attr_dice_list):.4f}')
    #     logging.info(f'{one_attr}-attr hd95 for Disc: {np.mean(one_attr_hd_list):.4f}')
    #     logging.info(f'{one_attr}-attr Jaccard/IoU for Disc: {np.mean(one_attr_jc_list):.4f}')

    logging.info('--------- Rim Performance for Attribute: {} -----------'.format(args.attribute))
    
     
    logging.info(f'Rim Overall Dice: {rim_overall_dice:.4f}')
    experiment_data['Rim Overall Dice'] = rim_overall_dice
    logging.info(f'Rim Overall hd95: {rim_overall_hd95:.4f}')
    logging.info(f'Rim Overall IoU: {rim_overall_jaccard:.4f}')
    experiment_data['Rim Overall IoU'] = rim_overall_jaccard


    es_rim_dice = equity_scaled_perf(dice_by_attr_rim, rim_overall_dice, no_of_attr)
    es_std_rim_dice = equity_scaled_std_perf(dice_by_attr_rim, rim_overall_dice, no_of_attr)
    es_rim_iou = equity_scaled_perf(jc_by_attr_rim, rim_overall_jaccard, no_of_attr)
    es_std_rim_iou = equity_scaled_std_perf(jc_by_attr_rim, rim_overall_jaccard, no_of_attr)

    logging.info(f'Rim Es-Dice: {es_rim_dice:.4f}')
    
    logging.info(f'Rim Es-IoU: {es_rim_iou:.4f}')
    

    logging.info(f'Rim Es-Std-Dice: {es_std_rim_dice:.4f}')
    experiment_data['Rim Es-Dice'] = es_std_rim_dice
    logging.info(f'Rim Es-Std-IoU: {es_std_rim_iou:.4f}')
    experiment_data['Rim Es-IoU'] = es_std_rim_iou


    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_rim[one_attr]
        one_attr_hd_list = hd_by_attr_rim[one_attr]
        one_attr_jc_list = jc_by_attr_rim[one_attr]

        logging.info(f'{one_attr}-attr dice for Rim: {np.mean(one_attr_dice_list):.4f}')
        experiment_data[f'{one_attr}-attr dice for Rim'] = np.mean(one_attr_dice_list)
        logging.info(f'{one_attr}-attr hd95 for Rim: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr Jaccard/IoU for Rim: {np.mean(one_attr_jc_list):.4f}')
        experiment_data[f'{one_attr}-attr IoU for Rim'] = np.mean(one_attr_jc_list)
   
    logging.info('------------------------------------------------------')
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jaccard : %f, mean_auc : %f, mean_cdr_distance : %f ' \
                 % (performance, mean_hd95, mean_jaccard, mean_auc, mean_cdr_dist))
    logging.info("Testing Finished!")
    
    # Example dictionary to hold your data

#     experiment_data = {
#     'Attribute': args.attribute,
#     'Overall Dice': np.mean(one_attr_dice_list),
#     'Overall HD95': np.mean(one_attr_hd_list),
#     'Overall IoU': np.mean(one_attr_jc_list),
#     'Cup Overall Dice': cup_overall_dice,
#     'Cup Overall HD95': cup_overall_hd95,
#     'Cup Overall IoU': cup_overall_jaccard,
#     'Rim Overall Dice': rim_overall_dice,
#     'Rim Overall HD95': rim_overall_hd95,
#     'Rim Overall IoU': rim_overall_jaccard,
#     'Mean Dice': performance,
#     'Mean HD95': mean_hd95,
#     'Mean Jaccard': mean_jaccard,
#     'Mean AUC': mean_auc,
#     'Mean CDR Distance': mean_cdr_dist,
#     'Cup Es-Dice': es_cup_dice,
#     'Cup Es-IoU': es_cup_iou,
#     'Cup Es-Std-Dice': es_std_cup_dice,
#     'Cup Es-Std-IoU': es_std_cup_iou,
#     'Rim Es-Dice': es_rim_dice,
#     'Rim Es-IoU': es_rim_iou,
#     'Rim Es-Std-Dice': es_std_rim_dice,
#     'Rim Es-Std-IoU': es_std_rim_iou
# }
#     experiment_data = {
#     'Attribute': args.attribute,
#     'Cup Es-Dice': es_cup_dice,
#     'Cup Overall Dice': cup_overall_dice,
#     'Cup Es-IoU': es_cup_iou,
#     'Cup Overall IoU': cup_overall_jaccard,
# }
    # for one_attr in range(no_of_attr):
    #     experiment_data[f'Attr{one_attr} Dice'] = np.mean(dice_by_attr[one_attr])
    #     # experiment_data[f'Attr{one_attr} HD95'] = np.mean(hd_by_attr[one_attr])
    #     experiment_data[f'Attr{one_attr} IoU'] = np.mean(jc_by_attr[one_attr])
    #     # experiment_data[f'Attr{one_attr} AUC'] = np.mean(auc_by_attr[one_attr]) if auc_by_attr[one_attr] else 'N/A'
    #     # experiment_data[f'Attr{one_attr} CDR'] = np.mean(cdr_by_attr[one_attr]) if cdr_by_attr[one_attr] else 'N/A'

    # Append the data of the current experiment to the CSV
    # append_to_csv('/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/equal_scale_out/test_outputs/experiment_results.csv', experiment_data)


    # 获取上一级路径
    csv_path = args.output
    csv_parent_path = os.path.abspath(os.path.join(csv_path, os.pardir)) 
    csv_parent_parent_path = os.path.abspath(os.path.join(csv_parent_path, os.pardir)) 
    append_to_csv(os.path.join(csv_parent_parent_path, 'experiment_results.csv'), experiment_data)
    
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict

def append_to_csv(file_name, dict_data):
    formatted_dict_data = {key: f'{value:.4f}' if isinstance(value, float) else value for key, value in dict_data.items()}
    # Open the file in append mode, create a new file if it doesn't exist
    with open(file_name, 'a', newline='') as csvfile:
        # Define the fieldnames based on the keys of the dictionary
        fieldnames = formatted_dict_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if the file is empty to write headers
        csvfile.seek(0, 2)  # Move to the end of the file
        if csvfile.tell() == 0:  # Check if the file is empty
            writer.writeheader()  # Write headers only if file is empty
        
        # Write the data
        writer.writerow(formatted_dict_data)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--datadir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='FairSeg', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output', type=str, default='./results/FairSeg_Yan_Loss_regular_gamma1.0_tanh_race/')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--list_dir', type=str, default='./SAMed/lists/FairSeg_final', help='list dir')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--attribute', type=str, default='race', help='attribute labels')
    parser.add_argument('--center_crop_size', type=int, default=512, help='center croped image size | 512 for slo, 420 for oct fundus')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--epoch', type=str, default='69')
    parser.add_argument('--exp_name', type=str, default='real')


    args = parser.parse_args()
    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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
    dataset_name = args.dataset
    dataset_config = {
        'FairSeg': {
            'Dataset': FairSeg_dataset,
            'data_dir': args.datadir,
            'num_classes': args.num_classes,
        }
    }
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    # log_folder = os.path.join(args.output_dir, 'test')
    log_folder = args.output
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    if args.attribute == 'race' or args.attribute == 'language':
        no_of_attr = 3
    else:
        no_of_attr = 2
        
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path, no_of_attr)
