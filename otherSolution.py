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

# グラフ表示の調査結果
# 年齢が若いほど、生存できる割合が50%に近づいているように見える
# 若い = 体力があるからこのような結果になった？

df["Fare"] = df["Fare"].fillna(df.groupby(["Pclass", 'Sex'])["Fare"].transform("median"))

cut_Fare = pd.cut(df["Fare"],4)

df["Fare"] = LabelEncoder().fit_transform(cut_Fare)

df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0])

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

df["Title"] = LabelEncoder().fit_transform(df["Title"])

df["Family_size"] = df["SibSp"] + df["Parch"]+1

# "SibSp", "Parch"をDataFrameから削除
# カラムを削除する際にはaxis = 1をつける
df = df.drop(["SibSp","Parch"], axis = 1)

#"Family_size"ラベルエンコーディング
df.loc[ df["Family_size"] == 1, "Family_size"] = 0                            # 独り身
df.loc[(df["Family_size"] > 1) & (df["Family_size"] <= 4), "Family_size"] = 1  # 小家族 
df.loc[(df["Family_size"] > 4) & (df["Family_size"] <= 6), "Family_size"] = 2  # 中家族
df.loc[df["Family_size"]  > 6, "Family_size"] = 3

# 家族が多すぎると身動きが取りにくく、逆に独り身だと、家族とともに乗船した人より、生き残ろうと思う気持ちが弱かった?。
# つまり、家族数も生存に関係していそうだと予測できます。

df["Ticket"] = df.Ticket.str.split().apply(lambda x : 0 if x[:][-1] == "LINE" else x[:][-1])
df.Ticket = df.Ticket.values.astype("int64")

df["TopName"] = df["Name"].map(lambda name:name.split(",")[0].strip())

# #女性または子どもはTrue
df["W_C"] = ((df.Title == 0) | (df.Sex == 1))

# #女性または子ども以外はTrue
df["M"] = ~((df.Title == 0) | (df.Sex == 1))

family = df.groupby(["TopName", "Pclass"])["Survived"]

df["F_Total"] = family.transform(lambda s: s.fillna(0).count())
df["F_Total"] = df["F_Total"].mask(df["W_C"], (df["F_Total"] - 1), axis=0)
df["F_Total"] = df["F_Total"].mask(df["M"], (df["F_Total"] - 1), axis=0)

df["F_Survived"] = family.transform(lambda s: s.fillna(0).sum())
df["F_Survived"] = df["F_Survived"].mask(df["W_C"], df["F_Survived"] - df["Survived"].fillna(0), axis=0)
df["F_Survived"] = df["F_Survived"].mask(df["M"], df["F_Survived"] - df["Survived"].fillna(0), axis=0)

df["F_S_Suc"] = (df["F_Survived"] / df["F_Total"].replace(0, np.nan))
df["F_S_Suc"].fillna(-1, inplace = True)

s_df = df.groupby(["F_S_Suc", "W_C"])["Survived"].agg(["mean", "count"])
# #"F_S_Suc"の計算で使用した説明変数の削除
df.drop(["TopName", "W_C", "M", "F_Total","F_Survived"], axis = 1, inplace = True)

df["PassengerId"] = df.index
df.drop(["Name","Embarked","Title", "Cabin"], axis=1, inplace=True)


#ダミー変数化
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
df = pd.get_dummies(df, columns=["Pclass", "Fare"])

#"Ticket"のみ標準化
num_features = ["Ticket"]

for col in num_features:
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1, 1))

#元の形に戻す（train, testデータの形に）
train, test = df.loc[train.index], df.loc[test.index]
#学習用データ
x_train = train.drop(["PassengerId","Survived"], axis = 1)
y_train = train["Survived"]
train_names = x_train.columns
#テスト用データ
x_test = test.drop(["PassengerId","Survived"], axis = 1)

#決定木
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
#学習
decision_tree.fit(x_train, y_train)
#推論
y_pred = decision_tree.predict(x_train)

print("正解率：", accuracy_score(y_train, y_pred))

#提出データ1
y_pred = decision_tree.predict(x_test)

#説明変数の重要度をグラフで表示（決定木）
importances = pd.DataFrame(decision_tree.feature_importances_, index = train_names)
importances.sort_values(by = 0, inplace=True, ascending = False)
importances = importances.iloc[0:6,:] 

#xgboost
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
#パラメータ
params = {'colsample_bytree': 0.5, 
         'learning_rate': 0.1, 
         'max_depth': 3, 
         'subsample': 0.9, 
         "objective":"multi:softmax", 
         "num_class":2}
#学習
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=10)

#推論
y_pred_2 = bst.predict(dtrain)

print("正解率：",accuracy_score(y_train, y_pred_2))

#提出データ2
y_pred_2 = bst.predict(dtest)

#submit用のファイル１を作成(決定木)
submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":y_pred.astype(int).ravel()})
submit.to_csv("xgb.csv",index = False)

#submit用のファイル2を作成(xgboost)
submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":y_pred_2.astype(int).ravel()})
submit.to_csv("tree.csv",index = False)