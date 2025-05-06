# <editor-fold desc="Imports">
import os
import time
import torch
import warnings
from tqdm import tqdm

import nibabel as nib
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Slurm
import argparse
import random
import numpy as np
import json

# </editor-fold>

# <editor-fold desc="Clinical Data Preparation and Cleaning">

# Default dataframe cleaning
clinical_data_path = "/scratch/tgoedietdoebe/AI-project/data/labels_BinClass.csv"
clinical_data = pd.read_csv(clinical_data_path)
clinical_data.dropna(inplace=True)
clinical_data.reset_index(drop=True, inplace=True)
clinical_data['w8_responder'] = clinical_data['w8_responder'].map({'Yes': 1, 'No': 0})
clinical_data['Stage1TX'] = clinical_data['Stage1TX'].map({'SER': 1, 'PLA': 0})
subjects_to_remove = ['CU0058', 'CU0059', 'CU0060', 'CU0061', 'CU0068', 'CU0074']
filtered_clinical_data = clinical_data[~clinical_data['ProjectSpecificId'].isin(subjects_to_remove)]
filtered_clinical_data.reset_index(drop=True, inplace=True)
sertraline_df = filtered_clinical_data[filtered_clinical_data['Stage1TX'] == 1]


fmri_data_path = "/data/projects/depredict/repositories/EMBARC/data/data_bids/derivatives/_fmriprep/output/"
subject_ids = sertraline_df['ProjectSpecificId'].tolist()
fmri_files = {}
missing_subjects = []
for sub_id in subject_ids:
    file_pattern = os.path.join(fmri_data_path,
                                f"sub-{sub_id}/ses-1/func/sub-{sub_id}_ses-1_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    if os.path.exists(file_pattern):
        fmri_files[sub_id] = file_pattern
    else:
        missing_subjects.append(sub_id)
processed_df = sertraline_df[sertraline_df['ProjectSpecificId'].isin(fmri_files.keys())]
processed_df = processed_df.copy()
processed_df['fMRI_path'] = processed_df['ProjectSpecificId'].map(fmri_files)
# </editor-fold>

# <editor-fold desc="Data Classes & Functions">
class fMRIDataset(Dataset):
    def __init__(self, df, data_dict, transform=None):
        """
        df: DataFrame containing at least 'ProjectSpecificId' and 'w8_responder'
        data_dict: {subject_id: torch.Tensor} holding the entire fMRI data for each subject
        transform: any optional transform
        """
        self.data = df.reset_index(drop=True)
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        # Each subject has 180 timepoints => 180 samples per subject
        return len(self.data) * 180

    def __getitem__(self, idx):
        # Compute which subject and which time index
        subject_idx = idx // 180
        time_idx = idx % 180

        row = self.data.iloc[subject_idx]
        subject_id = row['ProjectSpecificId']
        label = row['w8_responder']

        # Instead of torch.load, we just retrieve from the in-memory dictionary
        fmri_tensor = self.data_dict[subject_id]
        # fmri_tensor shape: [180, D, H, W] or [180, 1, D, H, W], depending on how it was saved

        # Extract the single timepoint
        # If shape is [180, D, H, W], you unsqueeze channel -> [1, D, H, W]
        #img = fmri_tensor[time_idx]
        img = fmri_tensor.mean(dim=0)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, D, H, W] so that conv3D sees channel=1

        # Normalize: per-volume zero-mean, unit variance
        # eps = small number to avoid division by zero if std is 0
        eps = 1e-6
        img = (img - img.mean()) / (img.std() + eps)

        # Optional transform if needed
        if self.transform:
            img = self.transform(img)

        # Check for NaNs / Infs (debugging)
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"NaN/Inf detected in subject {subject_id}, time {time_idx}")
            img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        return img, torch.tensor(label, dtype=torch.long)

def preload_data(df, preprocessed_path):
    """
    Loads all .pt files specified in df['ProjectSpecificId'] into memory (RAM)
    and returns a dictionary: {subject_id: fmri_tensor, ...}.

    Each fmri_tensor is the full 4D data for that subject: [T, D, H, W] or [T, 1, D, H, W].
    """
    data_dict = {}
    # Use tqdm for a nice progress bar
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preloading data"):
        subject_id = row["ProjectSpecificId"]
        pt_path = os.path.join(preprocessed_path, f"{subject_id}.pt")

        if not os.path.exists(pt_path):
            # Handle missing files, skip or raise error
            print(f"Warning: {pt_path} not found. Skipping subject {subject_id}.")
            continue

        fmri_tensor = torch.load(pt_path)
        data_dict[subject_id] = fmri_tensor

    return data_dict

def preprocess_and_save(fmri_path, subject_id, save_dir="/scratch/tgoedietdoebe/AI-project/data/processed/"):
    try:
        print(f"Processing {subject_id}...")
        fmri_img = nib.load(fmri_path)
        fmri_data = fmri_img.get_fdata()  # Should be [97, 115, 97, 180] ideally

        if fmri_data.ndim != 4:
            print(f"❌ Skipping {subject_id}: Expected 4D fMRI data, got shape {fmri_data.shape}")
            return

        if fmri_data.shape[3] != 180:
            print(f" Warning: {subject_id} has unexpected timepoints: {fmri_data.shape}")

        # Convert to PyTorch tensor (reorder to [T, D, H, W])
        # fmri_tensor = torch.tensor(fmri_data).permute(3, 0, 1, 2) # 1.6GB
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32).permute(3, 0, 1, 2)

        # Save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{subject_id}.pt")
        torch.save(fmri_tensor, save_path)

        print(f"✅ Saved {subject_id} | Final tensor shape: {fmri_tensor.shape} | Path: {save_path}")

    except Exception as e:
        print(f"❌ Failed to preprocess {subject_id}: {e}")
# </editor-fold>

# <editor-fold desc="Training & Evaluation Loop">
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    warnings.filterwarnings("ignore", category=FutureWarning)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'epoch_time': [],
        'data_time': [],
        'gpu_time': [],
        'batch_losses': []
    }

    for epoch in range(num_epochs):
        print(f"\n🚀 Starting Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_data_time = 0
        total_gpu_time = 0
        start_epoch = time.time()
        batch_losses = []

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch {epoch+1}'):

            start_data = time.time()
            images, labels = images.to(device), labels.to(device)
            data_time = time.time() - start_data

            start_gpu = time.time()
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            gpu_time = time.time() - start_gpu

            # Accumulate times and loss
            total_data_time += data_time
            total_gpu_time += gpu_time
            total_loss += loss.item()
            batch_losses.append(loss.item())


            if batch_idx % 10 == 0:
                throughput = images.size(0) / (data_time + gpu_time)
                print(
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Total_Loss: {total_loss}"
                    f"Data: {data_time:.3f}s | GPU: {gpu_time:.3f}s | "
                    f"Thru: {throughput:.2f} samples/s | "
                    f"MemUsed: {torch.cuda.memory_allocated(device)/1e6:.1f}MB"
                )

        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_epoch

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epoch_time'].append(epoch_time)
        history['data_time'].append(total_data_time)
        history['gpu_time'].append(total_gpu_time)
        history['batch_losses'].append(batch_losses)

        print(f"\n✅ Epoch {epoch + 1} Completed")
        print(f"   Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   Total Time: {epoch_time:.2f}s | ⏱️ Data: {total_data_time:.2f}s | 🧠 GPU: {total_gpu_time:.2f}s")

    return history

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    return avg_val_loss, accuracy
# </editor-fold>

# <editor-fold desc="Model Definition">
class HighAccuracy3DCNN(nn.Module):
    def __init__(self):
        super(HighAccuracy3DCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.feature_dim = 64 * 6 * 7 * 6

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)  # [B, 8, ~48, ~57, ~48]
        x = self.conv2(x)  # [B, 16, ~24, ~28, ~24]
        x = self.conv3(x)  # [B, 32, ~12, ~14, ~12]
        x = self.conv4(x)  # [B, 64, ~6, ~7, ~6]

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No sigmoid here — use BCEWithLogitsLoss

        return x
# </editor-fold>

# <editor-fold desc="Main Execution">
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=1)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--save_model_path', type=str, default="cnn_model.pt")
#     args = parser.parse_args()
#
#     # Reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
#     random.seed(42)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     # Data Preparation
#     train_data, test_data = train_test_split(processed_df, test_size=0.2)
#     train_data, val_data = train_test_split(train_data, test_size=0.25)
#
#     data_dict = preload_data(processed_df, preprocessed_path="/scratch/tgoedietdoebe/AI-project/data/processed/")
#     train_dataset = fMRIDataset(train_data, data_dict=data_dict)
#     val_dataset = fMRIDataset(val_data, data_dict=data_dict)
#     test_dataset = fMRIDataset(test_data, data_dict=data_dict)
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#
#     # Model
#     model = HighAccuracy3DCNN()
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     if torch.cuda.is_available():
#         print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#     else:
#         print("CUDA not available. Using CPU.")
#
#     # Training
#     history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs)
#
#     # Save Model
#     torch.save(model.state_dict(), args.save_model_path)
#     print(f"✅ Model saved to {args.save_model_path}")
#
#     # Save Training History
#     with open("train_history.json", "w") as f:
#         json.dump(history, f)
# </editor-fold>

# <editor-fold desc="Non-Slurm Execution">
# Data Preperation
train_data, test_data = train_test_split(processed_df, test_size=0.2)
train_data, val_data = train_test_split(train_data, test_size=0.25)

data_dict = preload_data(processed_df, preprocessed_path="/scratch/tgoedietdoebe/AI-project/data/processed/")
train_dataset = fMRIDataset(train_data, data_dict=data_dict)
val_dataset = fMRIDataset(val_data, data_dict=data_dict)
test_dataset = fMRIDataset(test_data, data_dict=data_dict)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 16, pin_memory= True )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Preperation
model = HighAccuracy3DCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Current Device: {torch.cuda.current_device()}")
print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Training & Evaluation
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)

# </editor-fold>