import pandas as pd
import numpy as np

# Raw Data
filename = "Data/kc_house_data.csv"
# Split Data
filename_train = "Data/kc_train.csv"
filename_validation = "Data/kc_valid.csv"
filename_test = "Data/kc_test.csv"

# returns raw dataframe
def raw_df():
    df = pd.read_csv(filename)
    return df
    
def zscore(column, mu, sigma):
    return column.apply(lambda x: (x-mu)/sigma)
    
def normalize(column, minimum, maximum):
    return column.apply(lambda x: (x-minimum)/(maximum-minimum))
    

def train_valid_test_split_raw():
    df_train = pd.read_csv(filename_train)
    df_valid = pd.read_csv(filename_validation)
    df_test = pd.read_csv(filename_test)
    return (df_train, df_valid, df_test)
    
### Method is either "normal" or "zscore"
### Returns (x_train, x_valid, x_test, y_train, y_valid, y_test)
def train_valid_test_split_clean(method="normal"):
    # Data frames
    df_train = pd.read_csv(filename_train)
    df_valid = pd.read_csv(filename_validation)
    df_test = pd.read_csv(filename_test)
    # Input Data
    x_train = df_train.drop("price", axis=1)
    x_valid = df_valid.drop("price", axis=1)
    x_test = df_test.drop("price", axis=1)
    # Output data
    y_train = df_train.price
    y_valid = df_valid.price
    y_test = df_test.price
    
    if method == "zscore":
        func = zscore
        arg1 = pd.concat([x_train, x_valid]).mean()
        arg2 = pd.concat([x_train, x_valid]).std()
    else:
        func = normalize
        arg1 = pd.concat([x_train, x_valid]).min()
        arg2 = pd.concat([x_train, x_valid]).max()
        
    for col in x_train.columns:
        x_train[col] = func(x_train[col], arg1[col], arg2[col])
        x_valid[col] = func(x_valid[col], arg1[col], arg2[col])
        x_test[col] = func(x_test[col], arg1[col], arg2[col])
    
    return (x_train, x_valid, x_test, y_train, y_valid, y_test)
    
    
        
