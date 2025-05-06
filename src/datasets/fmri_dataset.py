import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class fMRIDataset(Dataset):
    def __init__(self, df, data_dict, transform=None):
        """
        df: DataFrame with columns: ['subject_part', 'label']
        data_dict: {subject_part_id: 3D torch.Tensor}
        """
        self.data = df.reset_index(drop=True)
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject_part = row['subject_part']
        label = row['label']

        image = self.data_dict[subject_part]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def preload_mean_images(df, mean_image_path):
    data_dict = {}
    part_paths = []

    for subject_id in df["ProjectSpecificId"]:
        for i in range(1, 5): 
            part_paths.append((f"{subject_id}_part{i}.pt", f"{subject_id}_part{i}"))

    for filename, part_id in tqdm(part_paths, desc="Preloading mean images"):
        full_path = os.path.join(mean_image_path, filename)
        tensor = torch.load(full_path).float()
        data_dict[part_id] = tensor

    return data_dict