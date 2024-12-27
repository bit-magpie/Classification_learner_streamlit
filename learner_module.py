import numpy as np
import sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve

trained_models = dict()

classification_algorithms = {
    "LR": {
        "long_name": "Logistic Regression",
        "function": sklearn.linear_model.LogisticRegression,
        "parameters": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 100
        }
    },
    "DTC": {
        "long_name": "Decision Tree Classifier",
        "function": sklearn.tree.DecisionTreeClassifier,
        "parameters": {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    },
    "RFC": {
        "long_name": "Random Forest Classifier",
        "function": sklearn.ensemble.RandomForestClassifier,
        "parameters": {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2
        }
    },
    "GBC": {
        "long_name": "Gradient Boosting Classifier",
        "function": sklearn.ensemble.GradientBoostingClassifier,
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0
        }
    },
    "KNN": {
        "long_name": "K-Nearest Neighbors Classifier",
        "function": sklearn.neighbors.KNeighborsClassifier,
        "parameters": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "p": 2
        }
    },
    "LSVM": {
        "long_name": "Linear Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": {
            "kernel": "linear",
            "C": 1.0,
            "max_iter": -1
        }
    },
    "Q-SVM": {
        "long_name": "Quadratic Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": {
            "kernel": "poly",
            "degree": 2,
            "C": 1.0,
            "gamma": "scale"
        }
    },
    "C-SVM": {
        "long_name": "Cubic Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": {
            "kernel": "poly",
            "degree": 3,
            "C": 1.0,
            "gamma": "scale"
        }
    },
    "RBF-SVM": {
        "long_name": "Radial Basis Function Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale"
        }
    },
    "GNB": {
        "long_name": "Gaussian Naive Bayes",
        "function": sklearn.naive_bayes.GaussianNB,
        "parameters": {
            "var_smoothing": 1e-9
        }
    },
    "MNB": {
        "long_name": "Multinomial Naive Bayes",
        "function": sklearn.naive_bayes.MultinomialNB,
        "parameters": {
            "alpha": 1.0,
            "fit_prior": True
        }
    },
    "MLP": {
        "long_name": "Multi-Layer Perceptron Classifier",
        "function": sklearn.neural_network.MLPClassifier,
        "parameters": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "learning_rate": "constant"
        }
    },
    # "KMC": {
    #     "long_name": "K-Means",
    #     "function": sklearn.cluster.KMeans,
    #     "parameters": {
    #         "n_clusters": 8,
    #         "init": "k-means++",
    #         "n_init": 10,
    #         "max_iter": 300
    #     }
    # },
    "GMM": {
        "long_name": "Gaussian Mixture Model",
        "function": sklearn.mixture.GaussianMixture,
        "parameters": {
            "n_components": 1,
            "covariance_type": "full",
            "init_params": "kmeans",
            "max_iter": 100
        }
    },
    "XGBC": {
        "long_name": "XGBoost Classifier",
        "function": XGBClassifier,
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.3,
            "max_depth": 6,
            "gamma": 0
        }
    },
}
   

class Learner:
    def __init__(self, name, model, params=None):
        self.name = name
        self.model = None
        self.params = params
        # self.train_data = None
        # self.test_data = None
        self.accuracy = None
        self.f1 = None
        self.c_matrix = None
        self.auc = None
        self.num_cls = 2
        
        self._load_model(model)
    
    # def load_data(self, X, y, shuffle=True, train_split=0.8):
    #     if shuffle:
    #         X, y = self._shuffle_set(X, y)
        
    #     n_train = int(len(y) * train_split)
    #     self.train_data = [X[:n_train], y[:n_train]]
    #     self.test_data = [X[n_train:], y[n_train:]]
        
    # def set_train_test(self, train, test):
    #     self.train_data = train
    #     self.test_data = test
    
    def train_model(self, train_data):        
        X, y = train_data
        self.model.fit(X, y)
    
    def eval_model(self, test_data):
        if self.model is not None:
            X, y = test_data
            y_preds = self.model.predict(X)
            
            self.accuracy = accuracy_score(y, y_preds)            
            self.f1 = f1_score(y, y_preds, average='weighted')
            self.c_matrix = confusion_matrix(y, y_preds)            
            fpr, tpr, _ = roc_curve(y, y_preds, pos_label=self.num_cls)
            self.auc = auc(fpr, tpr)
        
    # def _shuffle_set(self, X, y):
    #     idx = np.arange(len(y))
    #     np.random.shuffle(idx)        
    #     return X[idx], y[idx]
        
    def _load_model(self, model):
        self.model = model(**self.params)
    
    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(model={self.name!r})'
        