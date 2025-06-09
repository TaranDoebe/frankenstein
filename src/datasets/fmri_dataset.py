import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class fMRIDataset(Dataset):
    def __init__(
        self,
        df,
        data_dict,
        transform=None,
        augment=True,
        raw_timepoints: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.data_dict = data_dict
        self.transform = transform
        self.augment = augment
        self.raw_timepoints = raw_timepoints

        self.n_tp = 180

    def __len__(self):
        return len(self.df) * self.n_tp if self.raw_timepoints else len(self.df)

    def __getitem__(self, idx):
        if self.raw_timepoints:
            subj_idx = idx // self.n_tp
            t_idx = idx %  self.n_tp
        else:
            subj_idx = idx
            t_idx = None 

        row = self.df.iloc[subj_idx]
        sid = row['ProjectSpecificId']
        vol = self.data_dict[sid]

        if self.raw_timepoints:
            img = vol[t_idx].unsqueeze(0).clone() 
        else:
            img = vol

        eps = 1e-6
        img = (img - img.mean()) / (img.std() + eps)

        if self.transform and self.augment:
            img = self.transform(img)

        label = row['w8_responder']
        sample_id = sid

        return img, torch.tensor(label, dtype=torch.float32), sample_id

class fMRISequentialDataset(Dataset):
    def __init__(
        self,
        df,
        data_dict,
        transform=None,
        augment=True,
        sequence_length: int = 60 
    ):
        self.df = df.reset_index(drop=True)
        self.data_dict = data_dict
        self.transform = transform
        self.augment = augment
        self.sequence_length = sequence_length
        self.n_tp_in_raw_scan = 180 

    def __len__(self):
        return len(self.df)
        

    def __getitem__(self, idx):
        eps = 1e-6 

        row = self.df.iloc[idx]
        sid = row['ProjectSpecificId']
        

        vol4d = self.data_dict[sid]#.clone() 
        sequence_ids = torch.arange(0,self.n_tp_in_raw_scan, self.n_tp_in_raw_scan//self.sequence_length) 
        vol4d = vol4d[sequence_ids]

        normalized_vol4d = torch.empty_like(vol4d)
        for t in range(vol4d.shape[0]): 
            tp_img = vol4d[t]  
            mean = tp_img.mean()
            std = tp_img.std()
            normalized_vol4d[t] = (tp_img - mean) / (std + eps)
        
        img = normalized_vol4d 
        label = row['w8_responder'] 
        sample_id = sid

        return img, torch.tensor(label, dtype=torch.float32), sample_id

def preload_timepoints(df, pt_dir, mean):
    data = {}
    for sid in tqdm(df['ProjectSpecificId'], desc="Preloading timepoints"):
        if mean:
            path = os.path.join(pt_dir, f"{sid}_mean.pt")
        else: # individual_timepoints
            path = os.path.join(pt_dir, f"{sid}.pt")
        data[sid] = torch.load(path).float()
    return data


