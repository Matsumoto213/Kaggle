import matplotlib.style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import seaborn as sns  # グラフ表示用
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.discriminant_analysis
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# RandomForestClassifier と XGBClassifier と LGBMClassifier で説明変数に対して目的変ん数をどの程度説明できているかをみる
def print_feature_importance(df, columns, target_column, random_seed):
    x = df[columns]
    y = df[target_column]

    print("--- RandomForestClassifier")
    model = sklearn.ensemble.RandomForestClassifier(random_state=random_seed)
    model.fit(x, y)
    fti1 = model.feature_importances_
    for i, column in enumerate(columns):
        print('{:20s} : {:>.6f}'.format(column, fti1[i]))


    print("--- XGBClassifier")
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(x, y)
    fti2 = model.feature_importances_
    for i, column in enumerate(columns):
        print('{:20s} : {:>.6f}'.format(column, fti2[i]))


    print("--- LGBMClassifier")
    model = lgb.LGBMClassifier(random_state=random_seed)
    model.fit(x, y)
    fti3 = model.feature_importances_   
    for i, column in enumerate(columns):
        print('{:20s} : {:>.2f}'.format(column, fti3[i]))

    #--- 結果をplot
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1, title="RandomForestClassifier(Feature Importance)")
    ax2 = fig.add_subplot(3, 1, 2, title="XGBClassifier(Feature Importance)")
    ax3 = fig.add_subplot(3, 1, 3, title="LGBMClassifier(Feature Importance)")
    ax1.barh(columns, fti1)
    ax2.barh(columns, fti2)
    ax3.barh(columns, fti3)
    fig.tight_layout()
    plt.show()

# 実行
print_feature_importance(df_train, select_columns, target_column, random_seed)

