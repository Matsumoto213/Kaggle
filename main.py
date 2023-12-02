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

def normalization(df, name):
    df[name] = (df[name] - df[name].mean()) / df[name].std()

missing_value(df_train)  # trainデータ
missing_value(df_test)    # testデータ

df_train["Embarked"].fillna("S", inplace=True)
df_test["Fare"].fillna(df_test['Fare'].median(), inplace=True)

# SibSpとParchのダミー化
def dummy(df):
    df = pd.get_dummies(df, columns = [
        'Pclass',
        'Sex',
        #'SibSp',
        #'Parch',
        "Embarked",
    ])
    return df

df_train = dummy(df_train)
df_test = dummy(df_test)

select_columns = [
    "Age",
    "Age_na",
    "SibSp",
    "Parch",
    "Fare", 
    "Pclass_1",
    "Pclass_2",
    #"Pclass_3",  # dummy除外
    "Sex_male",
    #"Sex_female",  # dummy除外
    "Embarked_C",
    "Embarked_Q",
    #"Embarked_S",  # dummy除外
]

def create_models(random_seed):
    models = [
        #Ensemble Methods
        sklearn.ensemble.AdaBoostClassifier(random_state=random_seed),
        sklearn.ensemble.BaggingClassifier(random_state=random_seed),
        sklearn.ensemble.ExtraTreesClassifier(random_state=random_seed),
        sklearn.ensemble.GradientBoostingClassifier(random_state=random_seed),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed),

        #Gaussian Processes
        sklearn.gaussian_process.GaussianProcessClassifier(random_state=random_seed),

        #GLM
        sklearn.linear_model.LogisticRegressionCV(random_state=random_seed),
        sklearn.linear_model.RidgeClassifierCV(),

        #Navies Bayes
        sklearn.naive_bayes.BernoulliNB(),
        sklearn.naive_bayes.GaussianNB(),

        #Nearest Neighbor
        sklearn.neighbors.KNeighborsClassifier(),

        #Trees
        sklearn.tree.DecisionTreeClassifier(random_state=random_seed),
        sklearn.tree.ExtraTreeClassifier(random_state=random_seed),

        #Discriminant Analysis
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
        sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),

        #xgboost
        xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=random_seed),

        # light bgm
        lgb.LGBMClassifier(random_state=random_seed),
    ]
    return models


def fit(df, columns, target_columns, random_seed):
    x = df[columns].to_numpy()
    y = df[target_columns].to_numpy()

    # 交叉検証
    model_scores = {}

    kf = sklearn.model_selection.KFold(n_splits = 3, shuffle = True, random_state = random_seed)
    for train_idx, true_idx in kf.split(x, y):
        # 各学習データとテストデータ
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_true = x[true_idx]
        y_true = y[true_idx]

        # 各モデルごとに学習
        for model in create_models(random_seed):
            name = model.__class__.__name__
            if name not in model_scores:
                model_scores[name] = []

                model.fit(x_train,y_train)
                pred_y = model.predict(x_true)

                model_scores[name].append((
                    sklearn.metrics.accuracy_score(y_true, pred_y),
                    sklearn.metrics.precision_score(y_true, pred_y),
                    sklearn.metrics.recall_score(y_true, pred_y),
                    sklearn.metrics.f1_score(y_true, pred_y),
                ))

        accs = []
        for k, scores in model_scores.items():
            scores = np.mean(scores, axis=0)  # 平均値の計算

            # モデル毎の平均
            print("正解率 {:.3f}, 適合率 {:.3f}, 再現率 {:.3f}, F値 {:.3f} : {}".format(
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                k,
            ))
            accs.append(scores)
        accs = np.median(accs, axis=0)  # 中央値
        print("正解率 {:.3f}, 適合率 {:.3f}, 再現率 {:.3f}, F値 {:.3f}".format(accs[0], accs[1], accs[2], accs[3]))

# 実行
fit(df_train, select_columns, target_column, random_seed)
