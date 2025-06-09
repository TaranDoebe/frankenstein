import pandas as pd
import numpy as np
import ast # Used to safely evaluate strings containing Python literals

# --- Configuration ---
INPUT_FILENAME = 'Results.csv'  # <--- CHANGE THIS to your actual input file name
OUTPUT_FILENAME = 'patient_features.csv'
COLUMN_TO_PROCESS = 'patient_probabilities'

# --- Main Script ---
print(f"Starting analysis of '{INPUT_FILENAME}'...")

# Load the input CSV file
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Successfully loaded '{INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"--- ERROR ---")
    print(f"The file '{INPUT_FILENAME}' was not found.")
    print("Please make sure the file is in the same directory as the script or provide the full path.")
    exit()

# Verify the required column exists
if COLUMN_TO_PROCESS not in df.columns:
    print(f"--- ERROR ---")
    print(f"The required column '{COLUMN_TO_PROCESS}' was not found in your CSV.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

# This list will store the calculated features for each row
results_list = []

# Process each row in the DataFrame
for index, row in df.iterrows():
    # Get the string representation of the nested list
    list_as_string = row[COLUMN_TO_PROCESS]

    try:
        # Safely convert the string to a Python list of lists
        # Example: '[[1, 2], [3, 4]]' becomes [[1, 2], [3, 4]]
        nested_lists = ast.literal_eval(list_as_string)
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse the data in row {index}. Skipping.")
        continue
    
    # Convert to a NumPy array for efficient calculations
    # The shape will be (10, 5), representing 10 lists with 5 values each
    data_array = np.array(nested_lists)
    
    # Transpose the array to group data by patient
    # The shape becomes (5, 10), where each row contains all 10 probabilities for one patient
    patient_data_grouped = data_array.T
    
    # Dictionary to store features for the current row
    row_features = {}

    # Calculate features for each patient (there are 5)
    for i in range(patient_data_grouped.shape[0]): # Loop 5 times for 5 patients
        patient_number = i + 1
        patient_probabilities = patient_data_grouped[i]
        
        # Calculate max, mean, and min for the current patient
        max_val = np.max(patient_probabilities)
        mean_val = np.mean(patient_probabilities)
        min_val = np.min(patient_probabilities)
        
        # Add the calculated features to our dictionary with descriptive names
        row_features[f'p{patient_number}_max_prob'] = max_val
        row_features[f'p{patient_number}_mean_prob'] = mean_val
        row_features[f'p{patient_number}_min_prob'] = min_val
        
    # Add the completed dictionary of features for this row to our results list
    results_list.append(row_features)

# Create a new DataFrame from the list of dictionaries
output_df = pd.DataFrame(results_list)

# Save the new DataFrame to a CSV file
output_df.to_csv(OUTPUT_FILENAME, index=False)

print("\n--- Analysis Complete! ---")
print(f"Created a new file named '{OUTPUT_FILENAME}' with the following columns:")
print(output_df.columns.tolist())
print("\nFirst 5 rows of the new feature table:")
print(output_df.head())