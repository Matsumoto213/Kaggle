#使用するライブラリ
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# データの読み込み
train = pd.read_csv("kaggle/input/titanic/train.csv")
test = pd.read_csv("kaggle/input/titanic/test.csv")

#indexを"PassengerId"に設定
train = train.set_index("PassengerId")
test = test.set_index("PassengerId")

#train, testデータの結合
df = pd.concat([train, test], axis=0, sort=False)

#"Sex"ラベルエンコーディング
df["Sex"] = df["Sex"].map({"female":1, "male":0})

select_columns = [
    "Survived",
    "Pclass",
    # "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    # "Ticket",
    "Fare",
    # "Cabin",
    # "Embarked"
]

# 相関関係を調査
# fig, axs = plt.subplots(figsize=(10, 8))
# sns.heatmap(df[select_columns].corr(),annot=True)
# plt.show()

# 相関関係の調査結果
# 男性に比べて女性の生存率の方が高い傾向がありそう
# また、乗客の年齢と運賃が上がるにつれて、等級が上がり（Pclass3からPclass1に）、等級が上がるにつれて生存率が高くなることが予想できる
# 等級が上がるにつれて生存率が高くなることが予想できる

# "Embarked"ラベルエンコーディング
df["Embarked"] = df["Embarked"].map({"C":0, "Q":1, "S":2})

# "Embarked"欠損値を中央値で補完
df["Embarked"] = df["Embarked"].fillna(df.Embarked.median())

# "Pclass"と"Sex"でグループ分けした、"Age"の平均値で欠損値を補完
df["Age"] = df["Age"].fillna(df.groupby(["Pclass","Sex"])["Age"].transform("mean"))

cut_Age = pd.cut(df["Age"], 4)

df["Age"] = LabelEncoder().fit_transform(cut_Age)

#pandasからグラフ表示（割合）
cross_Age = pd.crosstab(df["Age"], df["Survived"], normalize='index')
cross_Age.plot.bar(figsize=(10, 5))
plt.show()

# グラフ表示の調査結果
# 年齢が若いほど、生存できる割合が50%に近づいているように見える