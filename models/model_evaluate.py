import xgboost
import lightgbm
import catboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def rf_classify(n_jobs):
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def rf_regression(n_jobs):
    model = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def lr_classify(n_jobs, penalty='l2'):
    model = LogisticRegression(class_weight='balanced', n_jobs=n_jobs, tol=0.0005, C=0.1,
                               max_iter=10000, penalty=penalty)
    return model


def lr_regression(n_jobs):
    model = LinearRegression(n_jobs=n_jobs)
    return model


def xgb_classify(n_jobs):
    # model = xgboost.XGBClassifier(max_depth=6, learning_rate="0.1",
    #                               n_estimators=600, verbosity=0, subsample=0.8,
    #                               colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=n_jobs)
    model = xgboost.XGBClassifier(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def xgb_regression(n_jobs):
    # model = xgboost.XGBRegressor(max_depth=6, learning_rate="0.1",
    #                              n_estimators=600, verbosity=0, subsample=0.8,
    #                              colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=n_jobs)
    model = xgboost.XGBRegressor(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def lgb_classify(n_jobs):
    model = lightgbm.LGBMClassifier(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def lgb_regression(n_jobs):
    model = lightgbm.LGBMRegressor(n_estimators=10, random_state=0, n_jobs=n_jobs)
    return model


def cat_classify(n_jobs):
    model = catboost.CatBoostClassifier(n_estimators=10, random_state=0)
    return model


def cat_regression(n_jobs):
    model = catboost.CatBoostRegressor(n_estimators=10, random_state=0)
    return model


def rf_pre_classify(n_jobs):
    model = RandomForestClassifier(n_estimators=600, max_depth=8, n_jobs=n_jobs, class_weight='balanced', random_state=42)
    return model


def rf_pre_regression(n_jobs):
    model = RandomForestRegressor(n_estimators=600, max_depth=8, n_jobs=n_jobs, random_state=42)
    return model


model_fuctions = {"lr_regression": lr_regression, "lr_classify": lr_classify, "rf_regression": rf_regression,
                  "rf_classify": rf_classify, "xgb_regression": xgb_regression, "xgb_classify": xgb_classify,
                  "lgb_regression": lgb_regression, "lgb_classify": lgb_classify,"cat_regression": cat_regression,
                  "cat_classify": cat_classify, "rf_pre_regression": rf_pre_regression, "rf_pre_classify": rf_pre_classify,}
