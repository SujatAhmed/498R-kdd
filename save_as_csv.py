import numpy as np
import pandas as pd
from collections import defaultdict
import os

#                   -------------- All inputs are here (save all csv files for train val and test) --------------

dataset_key= ''                # which dataset (key from the file_paths)
split= 'vali'                  # train / validation / test
features= 1

preds_filename =      ''                            # predictions npy file
folder_path = ''                                     # Where to save the CSV file


#      -------- define a class for extracting the feature names ---------
class DatasetManager:

    
    file_paths = {                                            # provide the data path to the actual datasets (CSV files)
        'ETTh1': '',
        'ETTh2': '',
        'ETTm1': '',
        'ETTm2': '',
        'electricity': '',
        'exchange_rate': '',
        'traffic': '',
        'weather': '',
    }
    
    @classmethod
    def get_columns_excluding_date(cls, dataset_key):

        # Check if the dataset key is valid
        if dataset_key not in cls.file_paths:
            raise ValueError(f"Dataset '{dataset_key}' not found in the file path mappings.")
        
        # Get the file path
        file_path = cls.file_paths[dataset_key]
        
        # Read the CSV and exclude 'date' column
        data = pd.read_csv(file_path)
        column_names = [col for col in data.columns if col.lower() != 'date']
        return column_names


# -------------------------------------------------------- Input the  files  -------------------------------------------------------------

predictions = np.load(preds_filename)

pred_len = 96
stride = 1                                             


index_sum = defaultdict(lambda: np.zeros(features))
index_count = defaultdict(int)


for i in range(predictions.shape[0]):
    for j in range(pred_len):
        global_index = i * stride + j
        index_sum[global_index] += predictions[i, j, :]
        index_count[global_index] += 1


max_index = max(index_sum.keys())
final_predictions = np.zeros((max_index + 1, features))

for index in index_sum:
    final_predictions[index, :] = index_sum[index] / index_count[index]

print("Final Predictions Shape:", final_predictions.shape)


# ---------- Saving as CSV files ------------

def save_predictions_to_csv(predictions, column_names, file_path):
    """
    Save predictions to a CSV file using specified column names.
    """
    df = pd.DataFrame(predictions, columns=column_names)
    df.to_csv(file_path, index=False)



file_path = os.path.join(folder_path,f'{split}_predictions.csv')


column_names = DatasetManager.get_columns_excluding_date(dataset_key)

save_predictions_to_csv(final_predictions, column_names, file_path)
