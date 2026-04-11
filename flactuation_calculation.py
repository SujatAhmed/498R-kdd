import os
import pandas as pd
import numpy as np

#           ----------------------- All inputs here --------------------

# Specify the directory containing the CSV files
directory_path= ''              # train, val, test prediction files path (on the original series)




dataset1_path = r'G:\MI LAB\BTC\close_price.csv'     #    Ground truth csv files (with date)                           


dataset= 'close_price'

# comment it if no slicing is needed

start, end = 96, 20000     # index ==>> (included, excluded)





################################################################################################################################

#   part1: Merge the train val test csv files and store it

all_files = os.listdir(directory_path)


train_file = [f for f in all_files if "train" in f.lower()][0]
validation_file = [f for f in all_files if "vali" in f.lower()][0]
test_file = [f for f in all_files if "test" in f.lower()][0]

train_df = pd.read_csv(os.path.join(directory_path, train_file))
validation_df = pd.read_csv(os.path.join(directory_path, validation_file))
test_df = pd.read_csv(os.path.join(directory_path, test_file))

final_df = pd.concat([train_df, validation_df, test_df], axis=0, ignore_index=True)

final_df.to_csv(os.path.join(directory_path, f"{dataset}_merged_predictions.csv"), index=False)          # change here

dataset2_path=  os.path.join(directory_path, f"{dataset}_merged_predictions.csv")                       # automatically read the saved dataset

print("CSV files merged successfully..........")




# part2: load the ground truths and prediction csv files

# df1 = pd.read_csv(dataset1_path)
df2 = pd.read_csv(dataset2_path)                   # complete prediction series df2



# part3: slice the ground truths if the predictions are less than ground truths

def process_csv(file_path, start=0, end=None):
    df = pd.read_csv(file_path)
    
    return df.iloc[start:end].reset_index(drop=True)


df1 = sliced_data = process_csv(dataset1_path, start, end)      # original series df1

df2 = process_csv(dataset2_path, start, end)                                            # <<<<<<<<<<<<<<<<<<<< remove (slicing pred series) !!!!

# part4: finally calculate the fluctuation

if 'date' not in df1.columns:
    raise ValueError("The first dataset must contain a 'date' column.")
    
date_column = df1['date']
df1_numeric = df1.drop(columns=['date'])

if not (df1_numeric.columns == df2.columns).all():
    raise ValueError("The numeric columns of the datasets must match for subtraction.")

print("original series length: ", len(df1_numeric))
print("predicted series length: ", len(df2))

if len(df1_numeric) == len(df2):
    # fluctuation_numeric = df1_numeric - df2
    fluctuation_numeric = df2 - df1_numeric
    
    print("Datasets are of equal length. Skipping moving average and extension.")

else:
    moving_avg = df2.tail(125).mean()
    df2_extended = pd.concat(
        [df2, pd.DataFrame([moving_avg] * (len(df1_numeric) - len(df2)), columns=df2.columns)],
        ignore_index=True
    )

    assert len(df1_numeric) == len(df2_extended), "Length mismatch after extending df2."
    print("Datasets had different lengths. Extended df2 with moving average values.")

    # fluctuation_numeric = df1_numeric - df2_extended
    fluctuation_numeric = df2_extended - df1_numeric


fluctuation_df = pd.concat([date_column, fluctuation_numeric], axis=1)


fluctuation_df.to_csv(os.path.join(directory_path, f"{dataset}.csv"), index=False)                  

print("Fluctuation calculation completed...............'")

