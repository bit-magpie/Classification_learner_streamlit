import pandas as pd
import numpy as np
from itertools import compress
import sklearn.datasets as skds

data_file = None
dataset = None

sk_datasets = {"Iris": skds.load_iris, "Wine": skds.load_wine, "Digits": skds.load_digits, "Cancer": skds.load_breast_cancer}

class DataFile:
    def __init__(self, upload_file=None, df=None, name=None):
        self.df = None
        self.name = "Unknown"
        self.features = None
        self.selection = []
        self.target = None
        self.c_names = None
        if upload_file is not None:
            self._get_info(upload_file)
            
        if df is not None:
            self.name = name
            self.df = df
            
    
    def _get_info(self, upload_file):
        self.df = pd.read_csv(upload_file)
        self.name = upload_file.name   
        
    def set_features(self):
        if len(self.selection) > 0:
            self.features = list(compress(self.df.columns, self.selection))
            
    def get_train_data(self):
        if self.target is not None and self.features is not None:
            self.df[['classes']] = self.df[[self.target]].apply(lambda col:pd.Categorical(col).codes)
            self.c_names = list(self.df[self.target].unique())
            X = self.df[self.features].to_numpy()
            y = self.df[['classes']].to_numpy().ravel()
            return X, y
        
    def get_process_data(self, shuffle=True, train_split=0.8):
        X, y = self.get_train_data()
        if shuffle:
            X, y = self._shuffle_set(X, y)
        
        n_train = int(len(y) * train_split)
        train_data = [X[:n_train], y[:n_train]]
        test_data = [X[n_train:], y[n_train:]]
        
        return train_data, test_data
        
    def _shuffle_set(self, X, y):
        idx = np.arange(len(y))
        np.random.shuffle(idx)        
        return X[idx], y[idx]