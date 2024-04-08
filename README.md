# What is the SDT-ARMA

Soft Decision Tree with ARMA features. The parameters are validated and the best params are fitted. Now, the training and testing procedures are end to end. The example experiments conducted on Weekly M4 datasets. You can apply our code to all time series data by using the code. 

# Example Datasets
  
- https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv
- https://www.investing.com/currencies/hkd-usd-historical-data
- https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset
  explanation of the M4 dataset please go to https://paperswithcode.com/dataset/m4

# Requirements
  1. Creating a Conda Environment: To create a Conda environment, you typically use the conda create command followed by the name of the environment and the packages you want to install. Here's how you can create a Conda environment named sdt-arma:
```bash
  conda create --name my_env
```
  2. Activating the Environment: After creating the environment, you need to activate it. You can activate the environment using the following command: 
```bash
  conda activate --name my_env
```
  3. Installing Packages from requirements.txt: A requirements.txt file typically contains a list of dependencies along with their versions. To install packages listed in the requirements.txt file, you can use the following command.
```bash
  conda install --file requirements.txt
```
 4. Verifying Installation: For a final step, you can verify that all dependencies have been installed correctly by running:
```bash
  conda list
```
     
 
# Feature Extraction
Sample fature extraction code to extract label related feature if you need. It is in feature_extraction.ipynb file


# To Run The Code
```bash
  python main.py --exp_name exp --data_path csv_data.csv  --label_name y --test_size 0.3 --depth 3 --lamda 1e-3 --lr 1e-2 --epochs 30 --date_column_name date
```

# Parameter Explanation
--exp_name to give experiment name  <br />
--data_path to choose dataset from any csv. <br />
--label_name to understand the column name of the label in the csv of your dataset, default label name is y. <br />
--test_size to allocate the test from the whole dataset as ratio. 0.3 means th3 30 % of the whole dataset  <br />
--depth soft decision tree depth <br />
--lamda to choose coefficient of the regularization term <br />
--lr to choose learning rate. <br />
--epochs to give number of training epochs <br />
--date_column_name to understand whether there is date column or not to drop it before data converting to tensor <br />

# Comparision with Other Models
If you want to run other state of the art models, you can use model_comparision.ipynb file

