from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from medical_dataset import MedicalDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datetime import datetime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr_type', type=str, help='attribute type')
    parser.add_argument('--name', type=str, help='specific attribute to train')
    args = parser.parse_args()

    # Configs
    resume_path = './models/control_sd15_seg.pth'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    # sd_locked = True
    sd_locked = False
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc

    attribute = args.attr_type + '/' + args.name
    files = '/DATA_EDS/datasets/MedicalImage/txt/' + attribute + '.txt'
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = f"log_{current_time}_{args.attr_type}_{args.name}"

    dataset = MedicalDataset(files)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=log_dir)


    # Train!
    trainer.fit(model, dataloader)
