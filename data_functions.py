import pandas as pd
import numpy as np
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from itertools import compress
import sklearn.datasets as skds

dataset = None

if "Dataset" in st.session_state:
    data_file = st.session_state["Dataset"]

sk_datasets = {"Iris": skds.load_iris, 
               "Wine": skds.load_wine, 
               "Digits": skds.load_digits, 
               "Cancer": skds.load_breast_cancer
            }

class DataFile:
    def __init__(self, upload_file=None, df=None, name=None):
        self.df = None
        self.name = "Unknown"
        self.features = None
        self.selection = []
        self.target = None
        self.c_names = None
        self.train_data = None
        self.test_data = None
        self._df = None
        if upload_file is not None:
            self._get_info(upload_file)
            
        if df is not None:
            self.name = name
            self.df = df
            self._df = df.copy()
            
    
    def _get_info(self, upload_file):
        self.df = pd.read_csv(upload_file)
        self.name = upload_file.name   
        
    def set_features(self):
        self.c_names = self.df[self.target].unique()
        col_names = list(self.df.columns)
        col_names.remove(self.target)
        if len(self.selection) > 0:
            self.features = list(compress(col_names, self.selection))
            
    def get_train_data(self):
        if self.target is not None and self.features is not None:
            self._df[['classes']] = self._df[[self.target]].apply(lambda col:pd.Categorical(col).codes)
            self.c_names = list(self._df[self.target].unique())
            X = self._df[self.features].to_numpy()
            y = self._df[['classes']].to_numpy().ravel()
            return X, y
        
    def get_process_data(self, shuffle=True, train_split=0.8):
        X, y = self.get_train_data()
        
        if shuffle:
            X, y = self._shuffle_set(X, y)
        
        n_train = int(len(y) * train_split)
        train_data = [X[:n_train], y[:n_train]]
        test_data = [X[n_train:], y[n_train:]]
        
        self.train_data = train_data
        self.test_data = test_data
        
    def _shuffle_set(self, X, y):
        idx = np.arange(len(y))
        np.random.shuffle(idx)        
        return X[idx], y[idx]
    
def get_tsne():
    if "Dataset" in st.session_state:
        data_file = st.session_state["Dataset"]
        X = data_file.df[data_file.features]
        y = data_file.df[data_file.target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                    max_iter=1000, random_state=76)
        X_embedded = tsne.fit_transform(X_scaled)

        tsne_df = pd.DataFrame(X_embedded, columns=['t-SNE1', 't-SNE2'])
        tsne_df['target'] = y.values 

        return tsne_df