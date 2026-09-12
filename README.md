# Implementation Guideline

**Installation:** pip install -r requirements.txt 

## Start training>>

**step1:** (Train the original model using actual datasets and save the npy files)

       Train the iTransformer using MSE loss and the original datasets
       This will save the npy file for train validation and test splits separately.

**Step2:** (Address the overlaping indices and convert the npy file to a single value per index CSV )
       
       run the save_as_csv.py file. The 'dataset_key' is the exact name
       of the datasets. 'split' denotes tran val and test data. 

**step3:** (Generate the residual datasets)

       Go to the fluctuation_calculation.py file. The 'directory_path' is the same 'folder_path' 
       The 'start' and 'end' indices discard those indices that don't
       have predictions. 

**Step3:** (Train the model on residual data and predict the residuals)
       
       Once we have our residual dataset, upload the datasets to the dataset
       folder and again train the model using the Huber loss.

**Step4:** (final performance = base predictions- alpha*residual predictions)
  
      Go to the final_performance.py file.
      'slice_index' adjust the indices of base value predictions and fluctuatuaion predictions.


**commands for training:**
(1) ETT Datasets

bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh
bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh
bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh

(2) Traffic
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

(3) Weather
bash ./scripts/multivariate_forecasting/Weather/iTransformer.sh

(4) Exchange Rate
bash ./scripts/multivariate_forecasting/Exchange/iTransformer.sh

(5) Electricity
bash ./scripts/multivariate_forecasting/ECL/iTransformer.sh


**Additional Info:** Using low config GPU may require clearing the cache 
rm -rf "${HOME}/Library/Caches/CocoaPods"

If you need to clear the storage:
rm -rf ./checkpoints/*
rm -rf ./npy_store/*
rm -rf ./predictions/*
rm -rf ./figures/*



> **📚 Cite our paper**

```bibtex
@inproceedings{biswas2026one,
  title={One Step Closer to Ground Truth: A Multi-Scale Residual-Aware Representation Learning Pipeline for Predicting Time Series Data},
  author={Biswas, Amrijit and Kamal, Mustafa and Krambroeckers, Robin and Elahi, MM Lutfe and Momen, Sifat and Mohammed, Nabeel and Rahman, Shafin},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={162--173},
  year={2026}
}

      
