"""Final predictions = base predictions - alpha * residual predictions, scored before and after.

Example:
    python final_performnace.py --base_pred ./predictions/ETTh1_96_96/test_pred_96.npy \
        --ground_truth ./predictions/ETTh1_96_96/test_true_96.npy \
        --residual_pred ./predictions/ETTh1_residual_96_96/test_pred_96.npy --alpha 0.05 --out_dir ./final/ETTh1
"""
import argparse
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description='Apply the residual correction and report MSE / MAE')
parser.add_argument('--base_pred', required=True, help='base model test predictions (test_pred_<P>.npy)')
parser.add_argument('--ground_truth', required=True, help='base model test ground truth (test_true_<P>.npy)')
parser.add_argument('--residual_pred', required=True, help='meta model test predictions (test_pred_<P>.npy)')
parser.add_argument('--alpha', type=float, default=0.05, help='correction strength')
parser.add_argument('--out_dir', default='.', help='where to save the output .npy files')
args = parser.parse_args()

# --------------------------------------------------------------------------------------------------------------------------------

# Load data
series1 = np.load(args.base_pred)           # Shape: (N, pred_len, variables)
series2 = np.load(args.residual_pred)       # Shape: (M, pred_len, variables)
ground_truth = np.load(args.ground_truth)   # Shape: (N, pred_len, variables)

print("Before alignment base prediction shape: ", series1.shape)
print("Before alignment residual prediction shape: ", series2.shape)

# Both test sets end on the same time step; drop the extra windows at the START of the base set
prediction_start_index = series1.shape[0] - series2.shape[0]
if prediction_start_index < 0:
    raise ValueError("The residual predictions have more windows than the base predictions: check the file paths.")

series1_aligned = series1[prediction_start_index:, :, :]
ground_truth_aligned = ground_truth[prediction_start_index:, :, :]

assert series1_aligned.shape == series2.shape, "Shapes must match after alignment!"

print("After alignment base prediction shape: ", series1_aligned.shape)
print("After alignment residual prediction shape: ", series2.shape)


mse = mean_squared_error(ground_truth_aligned.flatten(), series1_aligned.flatten())
mae = mean_absolute_error(ground_truth_aligned.flatten(), series1_aligned.flatten())

print("Before: ")
print(mse)
print(mae)


final_preds = series1_aligned - args.alpha * series2

os.makedirs(args.out_dir, exist_ok=True)
np.save(os.path.join(args.out_dir, 'final_preds_original_scale.npy'), final_preds)

mse = mean_squared_error(ground_truth_aligned.flatten(), final_preds.flatten())
mae = mean_absolute_error(ground_truth_aligned.flatten(), final_preds.flatten())

print("After: ")
print(mse)
print(mae)


# ------------------------------------ Scaling -------------------------------

scaler = StandardScaler()

origianl_trues_flat = ground_truth_aligned.flatten().reshape(-1, 1)
base_predictions_flat = series1_aligned.flatten().reshape(-1, 1)
final_predictions_flat = final_preds.flatten().reshape(-1, 1)

scaler.fit(origianl_trues_flat)

scaled_trues = scaler.transform(origianl_trues_flat)
scaled_base = scaler.transform(base_predictions_flat)
scaled_predictions = scaler.transform(final_predictions_flat)

np.save(os.path.join(args.out_dir, 'final_trues_scaled.npy'), scaled_trues)
np.save(os.path.join(args.out_dir, 'final_preds_scaled.npy'), scaled_predictions)

base_mse_scaled = mean_squared_error(scaled_trues, scaled_base)
base_mae_scaled = mean_absolute_error(scaled_trues, scaled_base)
final_mse_scaled = mean_squared_error(scaled_trues, scaled_predictions)
final_mae_scaled = mean_absolute_error(scaled_trues, scaled_predictions)

print(f"Base MSE after scaling: {base_mse_scaled}")
print(f"Base MAE after scaling: {base_mae_scaled}")
print(f"Final MSE after scaling: {final_mse_scaled}")
print(f"Final MAE after scaling: {final_mae_scaled}")
