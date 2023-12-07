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

# 年齢を4分割
cut_Age = pd.cut(df["Age"], 4)

# "Age"ラベルエンコーディング
df["Age"] = LabelEncoder().fit_transform(cut_Age)

#pandasからグラフ表示（割合）
# cross_Age = pd.crosstab(df["Age"], df["Survived"], normalize='index')
# cross_Age.plot.bar(figsize=(10, 5))
# plt.show()

# グラフ表示の調査結果
# 年齢が若いほど、生存できる割合が50%に近づいているように見える
# 若い = 体力があるからこのような結果になった？


df["Fare"] = df["Fare"].fillna(df.groupby(["Pclass", 'Sex'])["Fare"].transform("median"))

cut_Fare = pd.cut(df["Fare"],4)

# fig, axes = plt.subplots(figsize = (15,5))
# sns.countplot(x = cut_Fare, hue = "Survived", data = df)
# plt.show()

df["Fare"] = LabelEncoder().fit_transform(cut_Fare)

# cross_Age = pd.crosstab(df["Fare"], df["Survived"], normalize='index')
# cross_Age.plot.bar(figsize=(10, 5))
# plt.show()

df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0])


# "Cabin" ごとの"Survived"を確認
# print(df.groupby(df["Cabin"])["Survived"].agg(["mean", "count"]))

# Cabinについてはデータから分かるように、欠損値(687)が多すぎるため、データの特徴を
# 掴みにくいことが考えられるので、機械学習には使用しない

# "Cabin"ラベルエンコーディング
df["Cabin"] = LabelEncoder().fit_transform(df["Cabin"])

#敬称の種類確認
df["Title"] = df.Name.str.extract("([A-Za-z]+)\.", expand = False)

other = ["Rev","Dr","Major", "Col", "Capt","Jonkheer","Countess"]

df["Title"] = df["Title"].replace(["Ms", "Mlle","Mme","Lady"], "Miss")
df["Title"] = df["Title"].replace(["Countess","Dona"], "Mrs")
df["Title"] = df["Title"].replace(["Don","Sir"], "Mr")
df["Title"] = df["Title"].replace(other,"Other")

#敬称ごとの生存率を確認
# df.groupby("Title").mean()["Survived"]

# fig,axs = plt.subplots(figsize = (15,5))
# sns.countplot(x="Title", hue="Survived", data=df)
# sns.despine()

df["Title"] = LabelEncoder().fit_transform(df["Title"])

# cross_Age = pd.crosstab(df["Title"], df["Survived"], normalize='index')
# cross_Age.plot.bar(figsize = (10, 5))
# plt.show()

df["Family_size"] = df["SibSp"] + df["Parch"]+1

# "SibSp", "Parch"をDataFrameから削除
# カラムを削除する際にはaxis = 1をつける
df = df.drop(["SibSp","Parch"], axis = 1)

# fig, axs = plt.subplots(figsize = (15, 5))
# sns.countplot(x="Family_size", hue="Survived", data=df)
# sns.despine()

# plt.show()

#"Family_size"ラベルエンコーディング
df.loc[ df["Family_size"] == 1, "Family_size"] = 0                            # 独り身
df.loc[(df["Family_size"] > 1) & (df["Family_size"] <= 4), "Family_size"] = 1  # 小家族 
df.loc[(df["Family_size"] > 4) & (df["Family_size"] <= 6), "Family_size"] = 2  # 中家族
df.loc[df["Family_size"]  > 6, "Family_size"] = 3

#pandasからグラフ表示（割合）
# cross_Age = pd.crosstab(df["Family_size"], df["Survived"], normalize='index')
# cross_Age.plot.bar(figsize=(10, 5))
# plt.show()

# 家族が多すぎると身動きが取りにくく、逆に独り身だと、家族とともに乗船した人より、生き残ろうと思う気持ちが弱かった?。
# つまり、家族数も生存に関係していそうだと予測できます。

df["Ticket"] = df.Ticket.str.split().apply(lambda x : 0 if x[:][-1] == "LINE" else x[:][-1])
df.Ticket = df.Ticket.values.astype("int64")

# 3つの変数をグループ分けして生存率と、生と死を合わせた総人数を調査
s_mean = df.rename(columns = {"Survived" : "S_mean"})
s_count = df.rename(columns = {"Survived" : "S_count"})

# s_mean = s_mean.groupby(["Sex", "Age", "Family_size"]).mean()["S_mean"]
# s_count = s_count.groupby(["Sex", "Age", "Family_size"]).count()["S_count"]

# pd.concat([s_mean, s_count], axis=1)

# #4つの変数をグループ分けして生存率と、生と死を合わせた総人数を調査（男性）
# m_s_mean = df.rename(columns={"Survived" : "S_mean"})
# m_s_count = df.rename(columns={"Survived" : "S_count"})

# m_s_mean = m_s_mean.groupby(["Sex", "Age", "Family_size", "Pclass"]).mean().head(29)["S_mean"]
# m_s_count = m_s_count.groupby(["Sex", "Age", "Family_size", "Pclass"]).count().head(29)["S_count"]

# pd.concat([m_s_mean, m_s_count], axis=1)
print(df['Survived'].isnull().sum())
