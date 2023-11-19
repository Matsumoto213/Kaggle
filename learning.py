import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.discriminant_analysis
import xgboost as xgb
import lightgbm as lgb
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
