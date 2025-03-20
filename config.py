from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {
    'max_depth': range(1, 20),
    "min_samples_split": range(2, 30)
}

rf_params = {
    "max_depth": [8, 15, None],
    "max_features": [5, 7, "sqrt"],
    "min_samples_split": [15, 20],
    "n_estimators": [200, 300]
}

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 200]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500]
}

classifiers = [
    ('KNN', KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(), lightgbm_params)
]