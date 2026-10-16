"""Average overlapping window predictions into one value per time index and save as CSV.

Example:
    python save_as_csv.py --preds ./predictions/ETTh1_96_96/epoch3_train_pred_96.npy \
        --dataset_csv ./dataset/ETT-small/ETTh1.csv --split train --out_dir ./residual_work/ETTh1
"""
import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Overlap-average window predictions and save them as a CSV')
parser.add_argument('--preds', required=True, help='predictions .npy file, shape (windows, pred_len, variables)')
parser.add_argument('--dataset_csv', required=True, help='original dataset CSV (used for the column names)')
parser.add_argument('--split', required=True, choices=['train', 'vali', 'test'], help='which split the predictions belong to')
parser.add_argument('--out_dir', required=True, help='folder to save <split>_predictions.csv in')
parser.add_argument('--target', default='OT', help='target column; moved last, as the data loader does')
parser.add_argument('--stride', type=int, default=1, help='step between consecutive windows')
args = parser.parse_args()


# -------- column names in the order the data loader uses --------
def get_columns_excluding_date(file_path, target):
    columns = [col for col in pd.read_csv(file_path, nrows=0).columns if col.lower() != 'date']
    if target in columns:
        columns = [col for col in columns if col != target] + [target]
    return columns


# -------- average every time index over all windows that cover it --------
predictions = np.load(args.preds, mmap_mode='r')
num_windows, pred_len, features = predictions.shape
print(f"Loaded {args.preds}: windows={num_windows}, pred_len={pred_len}, features={features}")

length = (num_windows - 1) * args.stride + pred_len
index_sum = np.zeros((length, features))
index_count = np.zeros(length)

for start in range(0, num_windows, 512):
    block = np.asarray(predictions[start:start + 512], dtype=np.float64)
    for j in range(pred_len):
        idx = (start + np.arange(len(block))) * args.stride + j
        index_sum[idx] += block[:, j, :]
        index_count[idx] += 1

final_predictions = index_sum / index_count[:, None]
print("Final Predictions Shape:", final_predictions.shape)


# -------- save as CSV --------
column_names = get_columns_excluding_date(args.dataset_csv, args.target)
if len(column_names) != features:
    raise ValueError(f"{args.dataset_csv} has {len(column_names)} variables but the predictions have {features}")

os.makedirs(args.out_dir, exist_ok=True)
file_path = os.path.join(args.out_dir, f'{args.split}_predictions.csv')
pd.DataFrame(final_predictions, columns=column_names).to_csv(file_path, index=False)
print(f"Saved {file_path}")
