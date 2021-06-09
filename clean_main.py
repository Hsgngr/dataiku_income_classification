# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:08:33 2021

@author: Ege
"""
import pandas as pd
import numpy as np
from feature_engineering_clean import dataiku_preprocessing

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedBaggingClassifier

import lightgbm as lgb

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

census_data =  pd.read_csv('census_income_learn.csv', header=None)
census_data_test =  pd.read_csv('census_income_test.csv', header=None)


#Preprocess Data
pre= dataiku_preprocessing()
df,df_test = pre.name_categories(census_data,census_data_test)
X_train,X_test =pre.process_data(df, df_test)

#Feature Selection
y_train = X_train.pop('y')
y_test =  X_test.pop('y')
X_train_r, X_test_r = pre.select_features(X_train, y_train, X_test, y_test)

#Model Training
model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model8 = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})
model5= GradientBoostingClassifier()
model6 = lgb.LGBMClassifier()
estimators= [('tr',model1), ('kn', model2), ('rfc_1000',model8),('rfc',model4), ('gbc', model5), ('lgb',model6)]
