import pandas as pd
import numpy as np
from itertools import compress

data_file = None
dataset = None

class DataFile:
    def __init__(self, upload_file):
        self.df = None
        self.name = None
        self.features = None
        self.selection = []
        self.target = None
        self._get_info(upload_file)
    
    def _get_info(self, upload_file):
        self.df = pd.read_csv(upload_file)
        self.name = upload_file.name   
        
    def set_features(self):
        if len(self.selection) > 0:
            self.features = list(compress(self.df.columns, self.selection))
    

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.x_train = X
        self.y_train = y
        self.x_test = X
        self.y_test = y        
    
    def split(self, percentage):
        idx = np.arange(len(self.X))
        np.random.shuffle(idx)
        
        sX = self.X[idx]
        sy = self.y[idx]
        
        n_train = int(len(self.X) * percentage)
        self.x_train = sX[:n_train]
        self.y_train = sy[:n_train]
        self.x_test = sX[n_train:]
        self.y_test = sy[n_train:]
    
    def __len__(self):
        return len(self.y)
    
    def __repr__(self):
        cls = self.__class__.__name__
        n = len(self.y)
        num_features = self.X.shape[1]
        return f'{cls}(samples={n}, features={num_features})'