import numpy as np
import sklearn

classification_algorithms = {
    "LR": {
        "long_name": "Logistic Regression",
        "function": sklearn.linear_model.LogisticRegression,
        "parameters": ["penalty", "C", "solver", "max_iter"]
    },
    "DTC": {
        "long_name": "Decision Tree Classifier",
        "function": sklearn.tree.DecisionTreeClassifier,
        "parameters": ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"]
    },
    "RFC": {
        "long_name": "Random Forest Classifier",
        "function": sklearn.ensemble.RandomForestClassifier,
        "parameters": ["n_estimators", "criterion", "max_depth", "min_samples_split"]
    },
    "GBC": {
        "long_name": "Gradient Boosting Classifier",
        "function": sklearn.ensemble.GradientBoostingClassifier,
        "parameters": ["n_estimators", "learning_rate", "max_depth", "subsample"]
    },
    # "XGBC": {
    #     "long_name": "XGBoost Classifier",
    #     "function": xgboost.XGBClassifier,
    #     "parameters": ["n_estimators", "learning_rate", "max_depth", "gamma"]
    # },
    "KNN": {
        "long_name": "K-Nearest Neighbors Classifier",
        "function": sklearn.neighbors.KNeighborsClassifier,
        "parameters": ["n_neighbors", "weights", "algorithm", "p"]
    },
    "LSVM": {
        "long_name": "Linear Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": ["kernel='linear'", "C", "max_iter"]
    },
    "Q-SVM": {
        "long_name": "Quadratic Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": ["kernel='poly'", "degree=2", "C", "gamma"]
    },
    "C-SVM": {
        "long_name": "Cubic Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": ["kernel='poly'", "degree=3", "C", "gamma"]
    },
    "RBF-SVM": {
        "long_name": "Radial Basis Function Support Vector Machine",
        "function": sklearn.svm.SVC,
        "parameters": ["kernel='rbf'", "C", "gamma"]
    },
    "GNB": {
        "long_name": "Gaussian Naive Bayes",
        "function": sklearn.naive_bayes.GaussianNB,
        "parameters": ["var_smoothing"]
    },
    "MNB": {
        "long_name": "Multinomial Naive Bayes",
        "function": sklearn.naive_bayes.MultinomialNB,
        "parameters": ["alpha", "fit_prior"]
    },
    "MLP": {
        "long_name": "Multi-Layer Perceptron Classifier",
        "function": sklearn.neural_network.MLPClassifier,
        "parameters": ["hidden_layer_sizes", "activation", "solver", "learning_rate"]
    },
    "KMC": {
        "long_name": "K-Means Clustering (used for classification via clustering)",
        "function": sklearn.cluster.KMeans,
        "parameters": ["n_clusters", "init", "n_init", "max_iter"]
    },
    "GMM": {
        "long_name": "Gaussian Mixture Model (used for classification via clustering)",
        "function": sklearn.mixture.GaussianMixture,
        "parameters": ["n_components", "covariance_type", "init_params", "max_iter"]
    }
}



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
        