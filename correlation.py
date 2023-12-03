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
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def print_correlation(df, columns):

    # 相関係数1:1
    print("--- 相関係数1:1 (threshold: 0.5)")
    cor = df[columns].corr()
    count = 0
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            val = cor[columns[i]][j]
            if abs(val) > 0.5:
                print("{} {}: {:.2f}".format(columns[i], columns[j], val))
                count += 1
    if count == 0:
        print("empty")
    print("")

    # heatmap
    plt.figure(figsize=(12,9))
    sns.heatmap(df[columns].corr(), annot=True, vmax=1, vmin=-1, fmt='.1f', cmap='RdBu')
    plt.show()


    # 相関係数1:多
    # 5以上だとあやしい、10だとかなり確定
    print("--- VIF(5以上だと怪しい)")
    vif = pd.DataFrame()
    x = df[columns]
    vif["VIF Factor"] = [
        variance_inflation_factor(x.values, i) for i in range(x.shape[1])
    ]
    vif["features"] = columns
    print(vif)
    plt.barh(columns, vif["VIF Factor"])
    plt.vlines([5], 0, len(columns), "blue", linestyles='dashed')
    plt.vlines([10], 0, len(columns), "red", linestyles='dashed')
    plt.title("VIF")
    plt.tight_layout()
    plt.show()

# 実行
print_correlation(df_train, select_columns)
