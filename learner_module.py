import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

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
    "XGBC": {
        "long_name": "XGBoost Classifier",
        "function": "xgboost.XGBClassifier",
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.3,
            "max_depth": 6,
            "gamma": 0
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
    "KMC": {
        "long_name": "K-Means Clustering (used for classification via clustering)",
        "function": sklearn.cluster.KMeans,
        "parameters": {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300
        }
    },
    "GMM": {
        "long_name": "Gaussian Mixture Model (used for classification via clustering)",
        "function": sklearn.mixture.GaussianMixture,
        "parameters": {
            "n_components": 1,
            "covariance_type": "full",
            "init_params": "kmeans",
            "max_iter": 100
        }
    }
}

def get_metrics(model, x_test, y_tests):
    y_preds = model.predict(x_test)
    return accuracy_score(y_tests, y_preds)
    
    

class Learner:
    def __init__(self):
        self.name = ""
        self.params = []
    
    def load_data(self):
        pass
    
    def train_model(self):
        pass
    
    def eval_model(self):
        pass
    
    def __repr__(self):
        pass
        