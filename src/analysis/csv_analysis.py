import pandas as pd
import os
import ast 

# Configuration: Adjust comments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'outputs', '06_07') # Change (arg-4) to match experiment directory
output_dir = DATA_DIR

map_directories = ['1', '2', '3', '4', '5', '6'] # Select target experiments
csv_filename = "outer_cv_results.csv"

output_filename = "Analysis_CNNGRU.csv" 
output_path = os.path.join(output_dir, output_filename)

# Execution

all_dataframes = []
for map_dir in map_directories:
    file_path = os.path.join(DATA_DIR, map_dir, csv_filename)
    df = pd.read_csv(file_path)
    all_dataframes.append(df)
    print(f"Loaded: {file_path}")

# - Selection

winning_rows_list = []
if all_dataframes:
    num_rows_to_compare = 20 # n=20 outer folds

    for i in range(num_rows_to_compare):
        best_row_for_index_i = None
        max_auc = -1.0
        source_map = None

        # Iterate through maps
        for map_index, df in enumerate(all_dataframes):
            current_map_name = map_directories[map_index]
            
            # Get the i-th row from the current dataframe
            current_row = df.iloc[i]
            current_auc = current_row['inner_mean_auc']

            # If this row has a better AUC, it's selected
            if current_auc > max_auc:
                max_auc = current_auc
                best_row_for_index_i = current_row.copy()
                source_map = current_map_name

        # New features 
        best_row_for_index_i['map'] = source_map
        patient_auc_list = ast.literal_eval(best_row_for_index_i['patient_auc'])

        max_patient_auc_val = max(patient_auc_list)
        best_row_for_index_i['max_patient_auc'] = max_patient_auc_val

        winning_rows_list.append(best_row_for_index_i)

# Save final DF
final_df = pd.DataFrame(winning_rows_list)
original_cols = [col for col in final_df.columns if col not in ['map', 'max_patient_auc']]
new_order = ['map', 'inner_mean_auc', 'max_patient_auc'] + original_cols
final_df = final_df[new_order]

os.makedirs(output_dir, exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print("\n Head DF:")
print(final_df.head())
