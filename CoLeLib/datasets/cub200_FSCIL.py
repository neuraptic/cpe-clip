import os

from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from PIL import Image
import pickle

class PKLDataset(Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.x = data['images']
        self.y = data['labels']
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx]).convert('RGB')
        x = self.transforms(img, return_tensors="pt").pixel_values[0]
        y = self.y[idx]

        return x, y

class CUB200FSCIL:
    def __init__(self, transforms, root="./splits/cub200"): 

        n_exps = 10
        self.num_classes_per_exp = [100] + [10]*n_exps
        self.classes_per_exp = [[i for i in range(100)]] + list(np.split(np.arange(100,200), n_exps))
        
        self.text_label_mapping = {v:k for k,v in pd.read_pickle(os.path.join(root, 'class_to_id.pkl')).items()}

        self.train_stream = [PKLDataset(pd.read_pickle(os.path.join(root, "base_class.pkl"))['train'], transforms)]
        self.eval_stream = [PKLDataset(pd.read_pickle(os.path.join(root, "base_class.pkl"))['test'], transforms)]
        for i in range(n_exps):
            data = pd.read_pickle(os.path.join(root, f"exp{i}.pkl"))
            train_data = data["train"]
            test_data = data["test"]
            self.train_stream.append(PKLDataset(train_data, transforms))
            self.eval_stream.append(PKLDataset(test_data, transforms))