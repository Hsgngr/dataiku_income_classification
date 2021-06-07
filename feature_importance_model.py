# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:33:18 2021

@author: Ege
"""
from sklearn.utils import resample
#Imbalance data
# Separate majority and minority classes
df_majority = df[df['y']==0]
df_minority = df[df['y']==1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=100000)
#Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=30000)
# Combine minority class with downsampled majority class
df_up_down_sampled = pd.concat([df_majority, df_minority_upsampled])


df_up_down_sampled.y.value_counts()

X_train = categorize_columns(nominal_columns,df= df_up_down_sampled, label_encoding=True)
y_train = X_train.pop('y')

model.fit(X_train,y_train)

model.score(X_test,y_test)
###############################################################################
# model used for feature importances
#Optional Feature Selection
from feature_selector import FeatureSelector

fs = FeatureSelector(data = X_train, labels = y_train)
fs.identify_collinear(correlation_threshold = 0.9)
collinear_features = fs.ops['collinear']

# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification',
                            eval_metric = 'binary_logloss',
                            n_iterations = 10,
                             early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']

# plot the feature importances
fs.plot_feature_importances(threshold = 0., plot_n = 20)

fs.identify_low_importance(cumulative_importance = 0.99)

# Remove the features from all methods (returns a df)
X_train_removed = fs.remove(methods = 'all')
X_train_removed, X_test = X_train_removed.align(X_test, axis=1, join='inner')

model.fit(X_train_removed,y_train)
model.score(X_test,y_test)

model = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90}, n_jobs =-1)
model.fit(X_train_removed,y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)


print(abs(y_test.value_counts()[0] - y_pred.value_counts()[0]), 'wrong predictions')

import joblib

joblib.dump(model,'rfc_model_1000_class_weight1_9.pkl')
joblib.dump(X_train_removed,'X_train.pkl')
joblib.dump(y_train,'y_train.pkl')

joblib.dump(X_test,'X_test.pkl')
joblib.dump(y_test,'y_test.pkl')

