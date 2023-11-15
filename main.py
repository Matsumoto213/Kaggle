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

def plot_category(df, column, target_column):
    # カウント情報
    print(pd.crosstab(df[column],df[target_column]))

    print("各クラス毎の生存率")
    print(pd.crosstab(df[column],df[target_column], normalize='index'))

    print("生存率に対する各クラスの割合")
    print(pd.crosstab(df[column],df[target_column], normalize='columns'))

    # plot
    sns.countplot(df[column], hue=df[target_column])
    plt.show()

plot_category(df_train, "Pclass", "Survived")
