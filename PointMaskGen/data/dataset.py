import os,sys
os.chdir(sys.path[0])
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np

synsetid_to_attribute = {
    '1': 'Hispanic', '2': 'non-Hispanic', '3': 'Unindentified', 
    '4': 'Female', '5': 'Male', 
    '6': 'English', '7': 'Other', '8': 'Spanish', '9': 'Unindentified', 
    '10': 'Divorced', '11': 'Legally_Separated', '12': 'Marriage_or_Partnered', '13': 'Not_Specified', '14': 'Single', '15': 'Widowed',
    '16': 'Asain', '17': 'Black', '18': 'White',
}
attribute_to_synsetid = {v: k for k, v in synsetid_to_attribute.items()} #名：数


class Fundus(Dataset):
    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, categories, attributes, scale_mode, shuffle, transform=None):
        super().__init__()
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34', 'no')
        self.path = path
        self.categories = categories
        self.attributes = attributes
        self.split = self.attributes
        self.attribute_synsetids = attribute_to_synsetid[attributes]
        self.scale_mode = scale_mode
        self.transform = transform
        self.shuffle = shuffle
        self.pointclouds = []


        self.get_statistics()
        self.load()

    def get_statistics(self):
        attribute = self.attributes
        dsetname = attribute
        data_dir = os.path.join(self.path, self.categories, dsetname)
        stats_dir = os.path.join(self.path, self.categories, dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_save_path = os.path.join(stats_dir, 'stats_' + dsetname + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = (torch.load(stats_save_path))
        else:
            pointclouds = []
            attribute_files = os.listdir(data_dir)
            for file in attribute_files:
                file = os.path.join(data_dir, file)
                pcd = np.loadtxt(file)
                #pcd = o3d.io.read_point_cloud(line)
                pointclouds.append(torch.from_numpy(np.array(pcd)).unsqueeze(0))
                #pointclouds.append(torch.from_numpy(np.array(pcd.points)).unsqueeze(0))
            all_points = torch.cat(pointclouds, dim=0)  # (B, N, 3)
            B, N, _ = all_points.size()
            mean = all_points.view(B * N, -1).mean(dim=0)  # (1, 3)
            std = all_points.view(-1).std(dim=0)  # (1, )
            self.stats = {'mean': mean, 'std': std}
            torch.save(self.stats, stats_save_path)

    def load(self):
        path_txt = os.path.join(self.path, self.categories, self.split + '.txt')

        def _enumerate_pointclouds(f=path_txt):
            idx = -1
            for line in open(f, 'r'):
                line = line.strip()
                pcd = np.loadtxt(os.path.join(self.path, self.categories, self.attributes, line))
                point = torch.from_numpy(np.array(pcd)).to(torch.float32)
                idx += 1
                yield point, idx
            #'global_unit', 'shape_unit', 'shape_half', 'shape_34', 'no'
        for pc, pc_id in _enumerate_pointclouds():
            if self.scale_mode == 'global_unit':
                shift = self.stats['mean'].reshape(1, 3)
                scale = self.stats['std'].reshape(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'no':
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            #scale =
            pc = ((pc - shift) / scale).to(torch.float32)

            self.pointclouds.append({
                'pointcloud': pc,
                'id': pc_id,
                'shift': shift,
                'scale': scale,
            })

            # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        if self.shuffle:
            random.Random(2020).shuffle(self.pointclouds)
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data
