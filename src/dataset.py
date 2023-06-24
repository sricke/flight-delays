import os
import typing as tp 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CAT_FEATURES = [
    #'Vlo-I',
    'DIANOM',
    'MES',
    'TIPOVUELO',
    'OPERA',
    'SIGLAORI',
    'SIGLADES'
]

SYNTHETIC_FEATURES = [
    'Temp-A',
    'Per-D',
    'ATRASADO'
]

LABEL = 'ATRASADO'
STRATIFY = 'OPERA'

# Dianom por sobre dia
# Numero operacion? Ya se tiene origen y destino. y diferencia con numero programado
# Ori-I = Ori-O = SIGLAORI
# ATRASADO = atraso + 15
# Temp-A por sobre mes y dia
# año solo 2 datos 2018


def load_dataset(
        filename: str,
        synthetic_filename: tp.Optional[tp.Union[str, bool]],
        val_size: float = 0.2,
        seed: int = 43
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    data_path = os.path.join("datasets", filename)
    synthetic_path = os.path.join("datasets", synthetic_filename) if synthetic_filename else None

    # load datasets
    dataset = pd.read_csv(data_path, usecols=CAT_FEATURES)
    synthetic_data = pd.read_csv(synthetic_path, usecols=SYNTHETIC_FEATURES) if synthetic_path else None

    # join them
    print(synthetic_data.head())
    dataset = pd.merge(dataset, synthetic_data, left_index=True, right_index=True) if synthetic_path else dataset

    # drop missing values. In this case only one
    dataset = dataset.dropna(axis=0) 

    # need to fix category type to train xgboost model
    dataset = dataset.astype("category")

    # get labels
    labels = dataset[LABEL]
    labels = LabelEncoder().fit_transform(labels)
    dataset = dataset.drop([LABEL], axis=1)
    print(dataset.head())
    # split dataset, stratified on airlines. 
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=val_size, random_state=seed, stratify=dataset[STRATIFY])

    print("---------- Dataset Info ----------")
    print("")
    print(f"Number of datapoints: {len(dataset)}")
    print(f"Number of datapoints train: {len(X_train)}")
    print(f"Percentage positive examples train: {np.sum(y_train)/len(y_train)}")
    print(f"Number of datapoints test: {len(X_test)}")
    print(f"Percentage positive examples test: {np.sum(y_test)/len(y_test)}")
    print(f"Number of features per row: {len(dataset.columns)}")
    return X_train, X_test, y_train, y_test

