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
# fit(df_train, select_columns, target_column, random_seed)

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
# print_feature_importance(df_train, select_columns, target_column, random_seed)
def print_statsmodels(df, columns, target_column):
    # 重回帰分析
    X1_train = sm.add_constant(df[columns])
    y = df[target_column]
    model = sm.OLS(y, X1_train)
    fitted = model.fit()

    # summary見方の参考
    # https://self-development.info/%E3%80%90%E5%88%9D%E5%BF%83%E8%80%85%E8%84%B1%E5%87%BA%E3%80%91statsmodels%E3%81%AB%E3%82%88%E3%82%8B%E9%87%8D%E5%9B%9E%E5%B8%B0%E5%88%86%E6%9E%90%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A6%8B%E6%96%B9/
    #print('summary = \n', fitted.summary())

    print("--- 重回帰分析の決定係数")
    for i, column in enumerate(columns):
        print('\t{:15s} : {:7.4f}(coef) {:5.1f}%(P>|t|)'.format(
            column, 
            fitted.params[i+1],
            fitted.pvalues[i]*100
        ))
    print("")

    # 各columnにおけるクック距離をだす
    print("--- 外れ値(cook_distance threshold:0.5)")
    for column in columns:
        # 単回帰分析
        X1_train = sm.add_constant(df[column])
        model = sm.OLS(y, X1_train)
        fitted = model.fit()

        cook_distance, p_value = OLSInfluence(fitted).cooks_distance
        kouho = np.where(cook_distance > 0.5)[0]
        if len(kouho) == 0:
            print("{:20s} cook_distance is 0(max: {:.4f})".format(column, np.max(cook_distance)))
        else:
            for index in kouho:
                print("{:20s} cook_distance: {}, index: {}".format(column, cook_distance[index], index))

    print("")

# 実行
# print_statsmodels(df_train, select_columns, target_column)

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
# print_correlation(df_train, select_columns)

# ------ start 提出用コード ------
# 学習データを作成
# x = df_train[select_columns].to_numpy()
# y = df_train[target_column].to_numpy()

# # 出力用データ
# x_test = df_test[select_columns].to_numpy()

# model = sklearn.linear_model.LogisticRegressionCV(random_state=random_seed)
# model.fit(x, y)
# pred_y = model.predict(x_test)

# # 提出用にデータ加工
# output = pd.DataFrame({'PassengerId': df_test["PassengerId"], 'Survived': pred_y})
# output.to_csv("LogisticRegressionCV.csv", header=True, index=False)
# print("Your submission was successfully saved!")
# ------ finish 提出用コード ------