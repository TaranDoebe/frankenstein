import os
import pandas as pd
import torch
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
    sertraline_df = main_clinical_df[main_clinical_df['Stage1TX'] == 1].copy()
    subject_ids = sertraline_df['ProjectSpecificId'].tolist()
    fmri_pt_files = {}
    missing_subjects = []

    for sub_id in subject_ids:
        file_path = os.path.join(pt_files_directory, f"{sub_id}.pt")
        if os.path.exists(file_path): fmri_pt_files[sub_id] = file_path
        else: missing_subjects.append(sub_id)

    processed_df = sertraline_df[sertraline_df['ProjectSpecificId'].isin(fmri_pt_files.keys())].copy()
    processed_df['fMRI_path'] = processed_df['ProjectSpecificId'].map(fmri_pt_files)
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df

def load_mean_parts_for_sertraline_subjects(base_clinical_df, mean_parts_dir, num_parts_expected=4):
    sertraline_subjects_df = base_clinical_df[base_clinical_df['Stage1TX'] == 1].copy()
    
    if sertraline_subjects_df.empty: return pd.DataFrame(columns=base_clinical_df.columns), {}

    valid_subject_ids_for_final_df = []
    loaded_tensor_data_dict = {}

    for subject_id in tqdm(sertraline_subjects_df['ProjectSpecificId'], desc="Processing SERTRALINE subjects"):
        current_subject_has_all_parts = True
        parts_loaded_for_current_subject = {}

        for i in range(1, num_parts_expected + 1):
            part_identifier = f"{subject_id}_part{i}"
            expected_part_file_path = os.path.join(mean_parts_dir, f"{part_identifier}.pt")
            
            if os.path.exists(expected_part_file_path):
                tensor_data = torch.load(expected_part_file_path)
                parts_loaded_for_current_subject[part_identifier] = tensor_data
        
        if current_subject_has_all_parts:
            valid_subject_ids_for_final_df.append(subject_id)
            loaded_tensor_data_dict.update(parts_loaded_for_current_subject)
        
    if not valid_subject_ids_for_final_df: return pd.DataFrame(columns=sertraline_subjects_df.columns), {}

    final_subjects_df = sertraline_subjects_df[sertraline_subjects_df['ProjectSpecificId'].isin(valid_subject_ids_for_final_df)].copy()
    final_subjects_df.reset_index(drop=True, inplace=True)
    
    num_subjects_with_parts = len(final_subjects_df)
    num_total_parts_loaded = len(loaded_tensor_data_dict)
    
    return final_subjects_df, loaded_tensor_data_dict

def create_subject_part_df(processed_df, data_dict):
    subject_part_list = []
    for part_id in data_dict.keys():
        subject_id_base = part_id.split("_part")[0]
        label_series = processed_df.loc[processed_df["ProjectSpecificId"] == subject_id_base, "w8_responder"]
        if not label_series.empty:
            label = int(label_series.values[0])
            subject_part_list.append({"subject_part": part_id, "label": label})

    return pd.DataFrame(subject_part_list)
