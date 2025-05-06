# utils/data_preparation.py

import os
import pandas as pd
import torch
# import nibabel as nib # No longer needed for this specific function if only loading .pt
from tqdm import tqdm

def load_and_clean_clinical_data(csv_path, subjects_to_remove):
    clinical_data = pd.read_csv(csv_path)
    clinical_data.dropna(inplace=True)
    clinical_data.reset_index(drop=True, inplace=True)
    clinical_data['w8_responder'] = clinical_data['w8_responder'].map({'Yes': 1, 'No': 0})
    clinical_data['Stage1TX'] = clinical_data['Stage1TX'].map({'SER': 1, 'PLA': 0})
    filtered_clinical_data = clinical_data[~clinical_data['ProjectSpecificId'].isin(subjects_to_remove)]
    filtered_clinical_data.reset_index(drop=True, inplace=True)
    return filtered_clinical_data

def get_sertraline_subject_pt_paths(main_clinical_df, pt_files_directory):
    """
    Filters for sertraline-treated subjects and finds paths to their .pt files.

    Args:
        main_clinical_df (pd.DataFrame): DataFrame containing clinical data including 'ProjectSpecificId' and 'Stage1TX'.
        pt_files_directory (str): The directory where the .pt files (named as 'ProjectSpecificId.pt') are stored.

    Returns:
        pd.DataFrame: A DataFrame filtered for sertraline subjects who have a corresponding .pt file,
                      with an added 'fMRI_path' column pointing to their .pt file.
    """
    sertraline_df = main_clinical_df[main_clinical_df['Stage1TX'] == 1].copy()
    subject_ids = sertraline_df['ProjectSpecificId'].tolist()
    fmri_pt_files = {}
    missing_subjects = []

    for sub_id in subject_ids:
        # Construct the expected .pt file path
        file_path = os.path.join(pt_files_directory, f"{sub_id}.pt")

        if os.path.exists(file_path):
            fmri_pt_files[sub_id] = file_path
        else:
            missing_subjects.append(sub_id)
            # Optional: print(f"Missing .pt file for subject: {sub_id} at {file_path}")

    if missing_subjects:
        print(f"Warning: Missing .pt files for {len(missing_subjects)} subjects: {missing_subjects[:5]}...") # Print first few

    # Filter the DataFrame to include only subjects for whom a .pt file was found
    processed_df = sertraline_df[sertraline_df['ProjectSpecificId'].isin(fmri_pt_files.keys())].copy()
    processed_df['fMRI_path'] = processed_df['ProjectSpecificId'].map(fmri_pt_files)
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df

def compute_and_normalize_mean_img(tensor, eps=1e-6):
    # This function assumes the input tensor is the raw time-series fMRI data
    # If your .pt files already store mean images, this function's usage context changes.
    mean_image = tensor.mean(dim=0) # Assuming time is the first dimension
    mean_image = (mean_image - mean_image.mean()) / (mean_image.std() + eps)
    return mean_image.unsqueeze(0) # Add channel dimension

def save_mean_image_parts(mean_images, subject_id, save_dir):
    for i, mean_image in enumerate(mean_images, start=1):
        save_path = os.path.join(save_dir, f"{subject_id}_part{i}.pt")
        torch.save(mean_image, save_path)

def precompute_and_save_mean_images(df_with_pt_paths, save_dir, min_timepoints=180, segment_timepoints=45):
    """
    Loads .pt files (assumed to be full fMRI time series), computes mean images for segments, and saves them.
    df_with_pt_paths: DataFrame with 'ProjectSpecificId' and 'fMRI_path' (pointing to subject.pt)
    """
    os.makedirs(save_dir, exist_ok=True)
    skipped_subjects_count = 0
    for index, row in tqdm(df_with_pt_paths.iterrows(), total=df_with_pt_paths.shape[0], desc="Precomputing mean images from .pt"):
        subject_id = row['ProjectSpecificId']
        fmri_pt_path = row['fMRI_path'] # This now points to subject_id.pt

        try:
            # Load the tensor from the .pt file
            # Ensure the loaded tensor has the shape (time, X, Y, Z) or similar
            fmri_tensor = torch.load(fmri_pt_path).float() # Assuming it's a 4D tensor T,D,H,W

            # Validate tensor dimensions (example, adjust as per your .pt file structure)
            if fmri_tensor.ndim < 4 or fmri_tensor.shape[0] < min_timepoints : # Assuming time is dim 0
                print(f"Skipping {subject_id}: Tensor shape {fmri_tensor.shape} not suitable or not enough timepoints ({fmri_tensor.shape[0]}).")
                skipped_subjects_count += 1
                continue

            num_segments = min_timepoints // segment_timepoints
            # Ensure segments are taken from the first 'min_timepoints' if more are available
            segments = [fmri_tensor[i*segment_timepoints:(i+1)*segment_timepoints] for i in range(num_segments)]
            mean_images = [compute_and_normalize_mean_img(segment) for segment in segments]

            save_mean_image_parts(mean_images, subject_id, save_dir)
        except Exception as e:
            print(f"Error processing {subject_id} from {fmri_pt_path}: {e}")
            skipped_subjects_count +=1

    if skipped_subjects_count > 0:
        print(f"Skipped {skipped_subjects_count} subjects during mean image precomputation from .pt files.")
    print(f"Mean images precomputation from .pt complete. Saved to: {save_dir}")


def create_subject_part_df(processed_df, data_dict):
    subject_part_list = []
    for part_id in data_dict.keys():
        subject_id_base = part_id.split("_part")[0]
        label_series = processed_df.loc[processed_df["ProjectSpecificId"] == subject_id_base, "w8_responder"]
        if not label_series.empty:
            label = int(label_series.values[0])
            subject_part_list.append({"subject_part": part_id, "label": label})
        else:
            print(f"Warning: Could not find label for {subject_id_base} from part_id {part_id}")
    return pd.DataFrame(subject_part_list)