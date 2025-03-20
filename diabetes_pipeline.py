################################################
# End-to-End Diabetes Machine Learning Pipeline II
################################################




################################################
# Helper Functions
################################################
from helpers import *



################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("/datasets/diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()

