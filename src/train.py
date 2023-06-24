import typing as tp
import os
import numpy as np
from dataset import load_dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

'''
Optimum params for training model.
Obtained from notebook/hyperparam_search
'''
params = {
    'subsample': 1.0,
    'n_estimators': 700,
    'min_child_weight': 10, 
    'max_depth': 6, 
    'learning_rate': 0.1, 
    'gamma': 0.2, 
    'colsample_bytree': 0.6
}

def train(
        params: tp.List,
        output_dir: str = 'models',
        seed: int = 43,
        val_size: float = 0.2
):
    # load the data
    X_train, X_test, y_train, y_test = load_dataset(
        filename = 'dataset_SCL.csv',
        synthetic_filename='synthetic_features.csv',
        seed=seed,
        val_size=val_size
    )
    print("---------- Model Info ----------")
    print("XGBoostClassifier")
    for param,number in params.items():
        print(f"{param}: {number}") 

    # in this case since we're only dealing with categorical columns and tree based method we dont need to preprocess input
    clf = XGBClassifier(
        **params,   
        tree_method="hist",
        eval_metric="error",
        enable_categorical=True, # needed since no preprocess
    )

    print("Training model...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)]) # train
    clf.save_model(os.path.join(output_dir, "flight_delay.json")) # save model

    # get probability scores on test
    y_score = clf.predict_proba(X_test)  
    # assign prediction
    y_pred = np.argmax(y_score, axis=1)
    # evaluate metrics
    auc = roc_auc_score(y_test, y_score[:, 1])
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("---------- Model Evaluation ----------")
    print("AUC:", auc)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    train(
        params = params,
        output_dir = 'models'
    )











    
