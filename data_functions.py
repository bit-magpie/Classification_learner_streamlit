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
        self.c_names = None
        self._get_info(upload_file)
    
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