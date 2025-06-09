import pandas as pd
import numpy as np

input_filename = 'LabelsCopy.csv'
output_filename = 'Labels_shuffled.csv'
label_column = 'w8_responder' 

df = pd.read_csv(input_filename)
labels_to_shuffle = df[label_column].to_numpy().copy()

np.random.seed(42)
np.random.shuffle(labels_to_shuffle)

df[label_column] = labels_to_shuffle
df.to_csv(output_filename, index=False)

print(f"\n{output_filename} = shuffled.")