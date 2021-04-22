#%%


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, recall_score

import warnings
# warnings.filterwarnings('ignore')


import sklearn
sklearn.__version__


def rf_tuning():
    df = pd.read_csv('../data/joined_cases_train.csv')
    df = df.drop(["latitude", "longitude", "Combined_Key", "country"], axis=1)
    y = df.pop('outcome')
    x_before = df

    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit_transform(df[['sex']])
    column_trans = make_column_transformer(
        (OneHotEncoder(), ['sex']),
        remainder='passthrough'
    )
    x = column_trans.fit_transform(x_before)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    scoring_list = {
        'f1': make_scorer(f1_score, average='weighted'),
        'Accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score, average='weighted')
    }
    grid_clf = GridSearchCV(RandomForestClassifier(bootstrap=True), {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 100]
        # 'min_samples_split': [2, 6, 10],
        # 'min_samples_leaf': [1, 5, 10],
        # 'max_leaf_nodes': [None, 200, 1000],
        # 'min_impurity_decrease': [0.0, 0.05, 0.1],
    }, scoring=scoring_list, refit='f1')



    grid_clf.fit(X_train, Y_train)
    print(grid_clf.cv_results_)

    # binarize the y label, get deceased metrics
    L = ["deceased","hospitalized","nonhospitalized","recovered"]
    binarized_y = label_binarize(Y_train, classes=["deceased","hospitalized","nonhospitalized","recovered"])
    Y_train_for_deceased = binarized_y[:, 0]

    print(set(Y_train_for_deceased))
    scoring_list2 = {
        'f1_deceased': make_scorer(f1_score, average='binary', pos_label=1),
        'Accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score)
    }
    grid_clf2 = GridSearchCV(RandomForestClassifier(bootstrap=True), {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 100],
        # 'min_samples_split': [2, 6, 10],
        # 'min_samples_leaf': [1, 5, 10],
        # 'max_leaf_nodes': [None, 200, 1000],
        # 'min_impurity_decrease': [0.0, 0.05, 0.1],
    }, scoring=scoring_list2, refit='f1_deceased')

    grid_clf2.fit(X_train, Y_train_for_deceased)

    print(grid_clf2.cv_results_)




if __name__ == '__main__':
    rf_tuning()
