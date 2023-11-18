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

# End テンプレート
df_train = pd.read_csv("kaggle/input/titanic/train.csv")
df_test = pd.read_csv("kaggle/input/titanic/test.csv")

target_column = "Survived"  # 目的変数
random_seed = 1234        # 乱数固定用

def missing_value(df):
    # 欠損値フラグ
    df["Age_na"] = df["Age"].isnull().astype(np.int64)

    # 欠損値を中央値で埋める
    df["Age"].fillna(df["Age"].median(), inplace = True)

print(df_train["Age_na"])