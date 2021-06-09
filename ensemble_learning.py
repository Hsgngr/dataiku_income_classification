# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 03:14:11 2021

@author: Ege
"""

# TensorFlow and tf.keras
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train_removed,y_train,validation_split=0.25, epochs=50)

###############################################################################
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

import lightgbm as lgb

model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()
model4= RandomForestClassifier()
model5= GradientBoostingClassifier()
model6 = lgb.LGBMClassifier()
model7 = make_pipeline(StandardScaler(),svm.LinearSVC(random_state=42))
model8 = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})
model9 = MLPClassifier(max_iter=300)


model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)
model5.fit(X_train,y_train)
model6.fit(X_train,y_train)
model7.fit(X_train,y_train)
model8.fit(X_train,y_train)
model9.fit(X_train,y_train)
###############################################################################
#VoteClassifier

estimators= [('tr',model1), ('kn', model2), ('lr',model3),('rfc',model4), ('gbc', model5), ('lgb',model6)]

vote_model = VotingClassifier(estimators=estimators, voting='hard')

vote_model.fit(X_train,y_train)
vote_model.score(X_test,y_test)

y_pred = vote_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

"""
[[93026   550]
 [ 3838  2348]]
              precision    recall  f1-score   support

           0       0.96      0.99      0.98     93576
           1       0.81      0.38      0.52      6186

    accuracy                           0.96     99762
   macro avg       0.89      0.69      0.75     99762
weighted avg       0.95      0.96      0.95     99762

0.9560153164531585
"""

###############################################################################
#Stacking
estimators= [('tr',model1), ('kn', model2), ('svr',model7),('rfc',model4), ('gbc', model5), ('lgb',model6)]


stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacked_model.fit(X_train,y_train)

y_pred = stacked_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
"""
[[92510  1066]
 [ 3211  2975]]
              precision    recall  f1-score   support

           0       0.97      0.99      0.98     93576
           1       0.74      0.48      0.58      6186

    accuracy                           0.96     99762
   macro avg       0.85      0.73      0.78     99762
weighted avg       0.95      0.96      0.95     99762

0.9571279645556424
"""
import joblib
joblib.dump(stacked_model,'stacked/stacked_model.pkl')
###############################################################################
#Stacking-v2
model8 = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})
estimators= [('tr',model1), ('kn', model2), ('rfc_1000',model8),('rfc',model4), ('gbc', model5), ('lgb',model6)]


stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacked_model.fit(X_train,y_train)

y_pred = stacked_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
"""
[[92559  1017]
 [ 3166  3020]]
              precision    recall  f1-score   support

           0       0.97      0.99      0.98     93576
           1       0.75      0.49      0.59      6186

    accuracy                           0.96     99762
   macro avg       0.86      0.74      0.78     99762
weighted avg       0.95      0.96      0.95     99762

0.958070207092881
"""
###############################################################################
#Stacking-v2
model8 = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})
estimators= [('tr',model1), ('kn', model2), ('rfc_1000',model8),('rfc',model4), ('gbc', model5), ('lgb',model6)]


stacked_model = StackingClassifier(estimators=estimators, final_estimator=model9)
stacked_model.fit(X_train,y_train)

y_pred = stacked_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
"""
[[92583   993]
 [ 3202  2984]]
              precision    recall  f1-score   support

           0       0.97      0.99      0.98     93576
           1       0.75      0.48      0.59      6186

    accuracy                           0.96     99762
   macro avg       0.86      0.74      0.78     99762
weighted avg       0.95      0.96      0.95     99762

0.9579499208115314
"""
###############################################################################
#Stacking-v3
model8 = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})
estimators= [('mlp',model9), ('kn', model2), ('rfc_1000',model8),('rfc',model4), ('gbc', model5), ('lgb',model6)]


stacked_model = StackingClassifier(estimators=estimators, final_estimator=model3,n_jobs = -1)
stacked_model.fit(X_train,y_train)

y_pred = stacked_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
"""
[[92583   993]
 [ 3202  2984]]
              precision    recall  f1-score   support

           0       0.97      0.99      0.98     93576
           1       0.75      0.48      0.59      6186

    accuracy                           0.96     99762
   macro avg       0.86      0.74      0.78     99762
weighted avg       0.95      0.96      0.95     99762

0.9579499208115314
"""
###############################################################################
pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3