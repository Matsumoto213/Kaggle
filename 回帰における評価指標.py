# 会期における評価指標
# RMSE (Root Mean Square Error : 平均平方2乗誤差)

# 各レコードの目的変数の芯の値と予測値の差の2乗を取り、それらを平均した後に
# 平方根を取ることで計算される。

#使用するライブラリ
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
import numpy as np

# y_trueが真の値、y_predが予測値
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.0, 1.5, 1.0, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)

# RMSLE(Root Mean Square Logarithmic Error)

# ①目的変数の対数をとって変換した値を新たな目的変数とした上でRMSEを最小化すれば、RMSLEを最小化することになる。
# ②目的変数が裾の重い分布を持ち、変換しないままだと大きな値の影響が強い場合や、真の値と予測値の比率に着目したい場合に用いられる。
# ③対数を取るにあたっては、真の値が0の時に値が負に発散するのを避けるため、通常は上記の通り1を加えてから対数をとる。
rmlsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

# MAE(Mean Absolute Error)

# ①MAEは外れ値の影響を提言した形での評価に適した関数
# ②yによる微分がyi = yiで不連続であったり、二次微分が常に0になってしまうという扱いづらい性質を持っている。
# ③仮に1つの代表ちで予測を行う場合、MAEを最小化する予測値は中央値。例えば、[1, 2, 3, 4, 10]と値がある場合、1点で予測したときに最もMAEが小さくなる予測値は中央値となる
ame = mean_absolute_error(y_true, y_pred)


# 決定係数

# 分母は予測値に依らず、分子は二乗誤差を差し引いているため、この指標を最大化することはRMSEを最小化することと同じ意味。
r2 = r2_score(y_true, y_pred)

# 混同行列

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

# 正例と予測して正例（真陽性）
tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))

# 負例と予測して負例（真陰性）
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))

# 負例と予測して正例（偽陽性）
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))

# 正例と予測して負例（偽陽性）
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp], [fn, tn]])
confusion_matrix2 = confusion_matrix(y_true,y_pred)

accuracy = accuracy_score(y_true, y_true)
