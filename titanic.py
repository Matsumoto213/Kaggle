#使用するライブラリ
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassfier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sklearn.ensemble
import itertools

# データの読み込み
train = pd.read_csv("kaggle/input/titanic/train.csv")
test = pd.read_csv("kaggle/input/titanic/test.csv")

train_x = train.drop(['Survived'], axis = 1)
test = train['Survived']

train_x = train.drop(['Survived'], axis = 1)
train_y = train['Survived']

test_x = test.copy()

train_x = train_x.drop(['PassengerId'], axis = 1)
test_x = test_x.drop(['PassengerId'], axis = 1)

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))


# モデルの作成および学習データを与えての学習
model = XGBClassfier(n_estimators = 28, random_state = 71)
model.fit(train_x, train_y)

# テストデータの予測値を確率で出力
pred = model.predict_proba(test_x)[:, 1]

# テストデータの予測値を二値に変換する
pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame({'PassengerId': test['PassengeId'], 'Servived': pred_label})
submission.to_csv('submission_first.csv', index = False)

scores_accuracy = []
scores_logloss = []
# クロスバリデーションを行う
# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す。
kf = KFold(n_splits = 4, shuffle = True, random_state = 71)
for tr_idx, va_idx in kf.split(train_x):
    # 学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    model = XGBClassfier(n_estimators = 20, random_state = 71)
    model.fit(tr_x, tr_y)

    va_pred = model.predict_proba(va_x)[:, 1]

    logloss = sklearn.metrics.log_loss(va_y, va_pred)
    accuracy = sklearn.metrics.acurracy_score(va_y, va_pred > 0.5)

    # そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)

print(f'logloss: {logloss:.4f}, accuracy: {accuracy: .4f}')

param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight' : [1.0, 2.0, 4.0]
}

param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

params = []
scores = []

for max_depth, min_child_weight in param_combinations:
    scores_fold = []

    kf = KFold(n_splits = 4, shuffle = True, random_state = 123456)