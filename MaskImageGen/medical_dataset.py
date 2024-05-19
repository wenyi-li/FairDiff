import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        self.data = []
        self.label = []
        with open(files, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                self.data.append(line)
                self.label.append(line.replace('images', 'labels'))
        assert len(self.data) == len(self.label), "Data len is not equal as label len."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_path = self.label[idx]

        source = cv2.resize(cv2.imread(label_path), (512,512))
        target = cv2.resize(cv2.imread(image_path), (512,512))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        prompt = "Scanning Laser Ophthalmoscope Fundus Image depicting the Optic Nerve Head and Central Cup"


        return dict(jpg=target, txt=prompt, hint=source)

class MedicalDatasetGenerate(Dataset):
    def __init__(self, dir):
        super().__init__()
        self.label = []
        files = os.listdir(dir)
        for file in files:
            self.label.append(os.path.join(os.path.abspath(dir), file))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_path = self.label[idx]

        source = cv2.resize(cv2.imread(label_path), (512,512))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        
        prompt = "Scanning Laser Ophthalmoscope Fundus Image depicting the Optic Nerve Head and Central Cup"

        return dict(txt=prompt, hint=source)
    
if __name__ == "__main__":

    files="/DATA_EDS2/liwy/datasets/MedicalImage/White.txt"
    dir = '/DATA_EDS2/liwy/datasets/MedicalImage/synthetic-label/race/Asian'

    dataset = MedicalDatasetGenerate(dir)
    print(len(dataset))

    item = dataset[1014]
    # jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    # print(jpg.shape)
    print(hint.shape)



