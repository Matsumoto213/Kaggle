
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
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
print_statsmodels(df_train, select_columns, target_column)
