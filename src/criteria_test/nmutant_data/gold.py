import sys
from sklearn.preprocessing import StandardScaler
import numpy as np    
import pandas as pd 
import os


def data_gold(datadir='res\\close_df.csv', train_test_split = 7000):
    current_path = os.path.dirname(__file__)
    print(os.getcwd())
    close_df = pd.read_csv(datadir,index_col=0)
    close_df.head()

    df = np.log(close_df).diff()
    target = df['XAU'].iloc[2:].values
    features = df.iloc[1:-1].values

    scaler = StandardScaler()

    X_train = scaler.fit_transform(features[:train_test_split])
    Y_train = target[:train_test_split].reshape(-1,1)

    X_test = scaler.transform(features[train_test_split:])
    Y_test = target[train_test_split:].reshape(-1,1)
    return X_train, Y_train, X_test, Y_test
