import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


file1_path =  ''                   # original series prediction files
ground_truth_path = ''             # original series ground truths

file2_path = ''                # residual series prediction files

alpha= 0.05

# --------------------------------------------------------------------------------------------------------------------------------

# Load data
series1 = np.load(file1_path)       # Shape: (N, 96, 321)
series2 = np.load(file2_path)       # Shape: (M, 96, 321)
ground_truth = np.load(ground_truth_path)  # Shape: (N, 96, 321)


print("Before alignment base prediction shape: ",series1.shape)
print("Before alignment residual prediction shape: ",series2.shape)

# Calculate how much to trim from the START of series1 and ground_truth
prediction_start_index = series1.shape[0] - series2.shape[0]  # = seq_len


# Trim the initial sequence (no predictions) from series1 and ground_truth
series1_aligned = series1[prediction_start_index:, :, :]      # Now shape: (M, 96, 321)
ground_truth_aligned = ground_truth[prediction_start_index:, :, :]  # Now shape: (M, 96, 321)


# Sanity check: Ensure shapes match for operations
assert series1_aligned.shape == series2.shape, "Shapes must match after alignment!"


print("After alignment base prediction shape: ",series1_aligned.shape)
print("After alignment residual prediction shape: ",series2.shape)


mse = mean_squared_error(ground_truth_aligned.flatten(), series1_aligned.flatten())
mae = mean_absolute_error(ground_truth_aligned.flatten(), series1_aligned.flatten())

print("Before: ")
print(mse)
print(mae)


final_preds= series1_aligned - (alpha) * (series2)

np.save('final_preds_original_scale.npy', final_preds)                                          # save 

mse = mean_squared_error(ground_truth_aligned.flatten(), final_preds.flatten())
mae = mean_absolute_error(ground_truth_aligned.flatten(), final_preds.flatten())

print("After: ")
print(mse)
print(mae)



# ------------------------------------ Scaling -------------------------------

scaler = StandardScaler()

origianl_trues_flat = ground_truth_aligned.flatten().reshape(-1, 1)
final_predictions_flat = final_preds.flatten().reshape(-1, 1)


scaler.fit(origianl_trues_flat)


scaled_trues = scaler.transform(origianl_trues_flat)
scaled_predictions = scaler.transform(final_predictions_flat)

np.save('final_trues_scaled.npy', scaled_trues)                                          # save 
np.save('final_preds_scaled.npy', scaled_predictions)                                          # save 


final_mse_scaled = mean_squared_error(scaled_trues, scaled_predictions)
final_mae_scaled = mean_absolute_error(scaled_trues, scaled_predictions)

print(f"Final MSE after scaling: {final_mse_scaled}")
print(f"Final MAE after scaling: {final_mae_scaled}")

