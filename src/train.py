import typing as tp
import os
from dataset import load_dataset
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

params = {
    "tree_method": "hist",
    "n_estimators": 32,
    "colsample_bylevel": 0.7,
}

def train(
        params: tp.List,
        output_dir: str = 'models',
        seed: int = 43,
        val_size: float = 0.2

):
    X_train, X_test, y_train, y_test = load_dataset(
        filename = 'dataset_SCL.csv',
        synthetic_filename='synthetic_features.csv',
        seed=seed,
        val_size=val_size
    )

    # in this case since we're only dealing with categorical columns and tree based method we dont need to preprocess input
    clf = XGBClassifier(
        **params,
        eval_metric="auc",
        enable_categorical=True, # needed since no preprocess
        max_cat_to_onehot=1,
    )

    print("Training model...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)])
    clf.save_model(os.path.join(output_dir, "flight_delay.json"))

    y_score = clf.predict_proba(X_test)[:, 1]  # proba of positive samples
    auc = roc_auc_score(y_test, y_score)
    print("AUC:", auc)


if __name__ == "__main__":
    train(
        params = params,
        output_dir = 'models'
    )











    
