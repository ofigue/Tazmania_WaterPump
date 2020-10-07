# Ref. https://github.com/abhishekkrthakur/mlframework

from sklearn import ensemble
from xgboost import XGBClassifier

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(),
    #"randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(),
    "XGBClassifier": XGBClassifier(learning_rate=0.1, objetive='multi:softmax', n_estimators=1000, max_depth=4,
    min_child_weight=6, subsample=0.8, colsample_bytree =0.8, nthread=4)
}