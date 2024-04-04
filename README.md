# What is the SDT-ARMA

Soft Decision Tree with ARMA features. The parameters are validated and the best params are fitted. Now, the training and testing procedures are end to end. The example experiments conducted on Weekly M4 datasets. You can apply our code to all time series data by using the code. 

# Example Datasets
  
- https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv
- https://www.investing.com/currencies/hkd-usd-historical-data
- https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset
  explanation of the M4 dataset please go to https://paperswithcode.com/dataset/m4

# Feature Extraction
Sample fature extraction code to extract label related feature if you need. It is in feature_extraction.ipynb file


# To run the code
```bash
     python main.py --datasets delhi --label_index --lr 0.3 --test_size 0.3 
```

# Parameter explanation
--datasets to choose dataset. 
--label_index to choose the index of the label in the csv of your dataset, usually the label is in the first or the last column.
--lr to choose learning rate.
--test_size to allocate the test from the whole dataset.
