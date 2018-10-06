## importing libraries
import numpy as np
import pandas as pd

import sys

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm

from sklearn import metrics

## class for Model
class Classification_Model:

    ## initialization function
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.X_train = pd.DataFrame.copy(X_train)
        self.X_test = pd.DataFrame.copy(X_test)
        
        self.y_train_raw = y_train
        self.y_test_raw = y_test
        self.y_train = pd.DataFrame.copy(y_train)
        self.y_test = pd.DataFrame.copy(y_test)

    ## Xgboost
    def runXGB(self):
        xgb_model = xgb.XGBClassifier()
        parameters = {
                'nthread':[4],
                'objective':['binary:logistic'],
                'learning_rate': [0.01,0.02,0.04],
                'max_depth': [3,5],
                'min_child_weight': [11],
                'silent': [1],
                'subsample': [0.8],
                'colsample_bytree': [0.7],
                'n_estimators': [300,400,500],
                'missing':[-999],
                'seed': [1337]
              }
        
        xgbclf_gs = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(self.y_train, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)
        xgbclf_gs.fit(self.X_train, self.y_train)
        
        best_parameters, score, _ = max(xgbclf_gs.grid_scores_, key=lambda x: x[1])
        print("Best Parameters : " % best_parameters)
        
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(self.y_train, xgbclf_gs.predict_proba(self.X_train)[:,1])
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(self.y_test, xgbclf_gs.predict_proba(self.X_test)[:,1])
        print("Training AUC : " % metrics.auc(fpr_train, tpr_train))
        print("Test AUC : " % metrics.auc(fpr_test, tpr_test))
        
        return xgbclf_gs
        
    # Random Forest
    def runRF(self):
        rf_model = RandomForestClassifier()
        parameters = {
                'bootstrap': [True],
                'max_depth': [5, 7, 9, 11],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [300, 500, 700, 900]
                }
        
        rfclf_gs = GridSearchCV(rf_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(self.y_train, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)
        rfclf_gs.fit(self.X_train, self.y_train)
        
        best_parameters, score, _ = max(rfclf_gs.grid_scores_, key=lambda x: x[1])
        print("Best Parameters : " % best_parameters)
        
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(self.y_train, rfclf_gs.predict_proba(self.X_train)[:,1])
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(self.y_test, rfclf_gs.predict_proba(self.X_test)[:,1])
        print("Training AUC : " % metrics.auc(fpr_train, tpr_train))
        print("Test AUC : " % metrics.auc(fpr_test, tpr_test))
        
        return rfclf_gs
    
    # Light GBM
    def runlgbm(self):
        lgbm_model = lgbm.LGBMClassifier()
        parameters = {
                'learning_rate': [0.01],
                'max_depth': [5, 7, 9, 11],
                'n_estimators': [200, 400, 600, 800],
                'num_leaves': [6,8,12,16],
                'boosting_type' : ['gbdt'],
                'objective' : ['binary'],
                'random_state' : [501],
                'colsample_bytree' : [0.65, 0.66],
                'subsample' : [0.7,0.75],
                'reg_alpha' : [1,1.2],
                'reg_lambda' : [1,1.2,1.4]
                }
        
        lgbm_gs = GridSearchCV(lgbm_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(self.y_train, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)
        lgbm_gs.fit(self.X_train, self.y_train)
        
        best_parameters, score, _ = max(lgbm_gs.grid_scores_, key=lambda x: x[1])
        print("Best Parameters : " % best_parameters)
        
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(self.y_train, lgbm_gs.predict_proba(self.X_train)[:,1])
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(self.y_test, lgbm_gs.predict_proba(self.X_test)[:,1])
        print("Training AUC : " % metrics.auc(fpr_train, tpr_train))
        print("Test AUC : " % metrics.auc(fpr_test, tpr_test))
        
        return lgbm_gs