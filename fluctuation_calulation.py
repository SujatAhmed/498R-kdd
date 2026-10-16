"""Merge the train/vali/test prediction CSVs and compute the residual (fluctuation) series.

Example:
    python flactuation_calculation.py --pred_dir ./residual_work/ETTh1 \
        --ground_truth ./dataset/ETT-small/ETTh1.csv --dataset ETTh1_residual --start 96 --end 14400
"""
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(description='Compute residuals = predictions - ground truth')
parser.add_argument('--pred_dir', required=True, help='folder with the train / vali / test prediction CSVs')
parser.add_argument('--ground_truth', required=True, help='original dataset CSV (with a date column)')
parser.add_argument('--dataset', required=True, help='name of the output files (<dataset>.csv)')
parser.add_argument('--start', type=int, default=96, help='first ground-truth row with a prediction (= seq_len)')
parser.add_argument('--end', type=int, default=None,
                    help='row after the last prediction (ETTh: 14400, ETTm: 57600, custom: file length)')
parser.add_argument('--out_dir', default=None, help='where to save the residual CSV (default: --pred_dir)')
parser.add_argument('--target', default='OT', help='target column; moved last, as the data loader does')
args = parser.parse_args()

out_dir = args.out_dir or args.pred_dir
os.makedirs(out_dir, exist_ok=True)


# part1: merge the train, val and test CSV files and store them
all_files = os.listdir(args.pred_dir)

def find_file(key):
    matches = [f for f in all_files if key in f.lower() and f.endswith('.csv')]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{key}' CSV in {args.pred_dir}, found {matches}")
    return matches[0]

train_df = pd.read_csv(os.path.join(args.pred_dir, find_file('train')))
validation_df = pd.read_csv(os.path.join(args.pred_dir, find_file('vali')))
test_df = pd.read_csv(os.path.join(args.pred_dir, find_file('test')))

df2 = pd.concat([train_df, validation_df, test_df], axis=0, ignore_index=True)    # complete prediction series
merged_path = os.path.join(out_dir, f"{args.dataset}_merged_predictions.csv")
df2.to_csv(merged_path, index=False)
print("CSV files merged successfully..........")


# part2: load the ground truth and keep only the rows that have predictions
df1 = pd.read_csv(args.ground_truth).iloc[args.start:args.end].reset_index(drop=True)

if 'date' not in df1.columns:
    raise ValueError("The ground-truth dataset must contain a 'date' column.")

date_column = df1['date']
df1_numeric = df1.drop(columns=['date'])
if args.target in df1_numeric.columns:
    df1_numeric = df1_numeric[[c for c in df1_numeric.columns if c != args.target] + [args.target]]

if list(df1_numeric.columns) != list(df2.columns):
    raise ValueError("The numeric columns of the datasets must match for subtraction.")

print("original series length: ", len(df1_numeric))
print("predicted series length: ", len(df2))


# part3: calculate the fluctuation (residual = prediction - ground truth)
if len(df1_numeric) == len(df2):
    print("Datasets are of equal length. Skipping moving average and extension.")
    fluctuation_numeric = df2 - df1_numeric
else:
    if len(df2) > len(df1_numeric):
        raise ValueError("More predictions than ground-truth rows: check --start / --end.")
    print("WARNING: predictions are shorter than the ground truth; extending them with a moving average. "
          "This usually means --start / --end are wrong.")
    moving_avg = df2.tail(125).mean()
    df2_extended = pd.concat(
        [df2, pd.DataFrame([moving_avg] * (len(df1_numeric) - len(df2)), columns=df2.columns)],
        ignore_index=True
    )
    fluctuation_numeric = df2_extended - df1_numeric

fluctuation_df = pd.concat([date_column, fluctuation_numeric], axis=1)
out_path = os.path.join(out_dir, f"{args.dataset}.csv")
fluctuation_df.to_csv(out_path, index=False)
print(f"Fluctuation calculation completed: {out_path}  shape={fluctuation_df.shape}")
