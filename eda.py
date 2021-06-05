# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:39:59 2021

@author: Ege
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


census_data =  pd.read_csv('census_income_learn.csv', header=None)
census_data_test =  pd.read_csv('census_income_test.csv', header=None)

#Add the column names for each column
column_names_dict={
    0 : 'age',
    1 : 'class_of_work',
    2 : 'industry_code',
    3 : 'occupation_code',
    4 : 'education',
    5 : 'wage_per_hour',
    6 : 'enrolled_in_edu_inst_last_wk',
    7 : 'marital_status',
    8 : 'major_industry_code',
    9: 'major_occupation_code',
    10: 'race',
    11: 'hispanic_origin',
    12: 'sex',
    13: 'member_of_labor_union',
    14: 'reason_for_unemployment',
    15: 'full_or_part_time_employment_stat',
    16: 'capital_gains',
    17: 'capital_losses',
    18: 'dividends_from_stocks',
    19: 'tax_filer_status',
    20: 'region of previous residence',
    21: 'state_of_previous_residence',
    22: 'detailed_household_and_family_stat',
    23: 'detailed_household_summary_in_household',
    24: 'instance_weight',
    25: 'migration_code_change_in_msa',
    26: 'migration_code_change_in_reg',
    27: 'migration_code_move_within_reg',
    28: 'live_in_this_house_1_year_ago',
    29: 'migration_prev_res_in_sunbelt',
    30: 'num_persons_worked_for_employer',
    31: 'family_member_under_18',
    32: 'country_of_birth_father',
    33: 'country_of_birth_mother',
    34: 'country_of_birth_self',
    35: 'citizenship',
    36: 'own_business_or_self_employed',
    37: 'fill_inc_questionnaire_for_veterans_admin',
    38: 'veteran_benefits',
    39: 'weeks_worked_in_years',
    40: 'year',
    41: 'y'    
    }

df = census_data.head(10)
df.rename(columns= column_names_dict, inplace =True)

df = census_data.rename(columns = column_names_dict)
df_test = census_data_test.rename(columns = column_names_dict)
###############################################################################

def categorize_columns(columns, df=df, label_encoding =False):
    temp = df.copy()
    for column in columns:
        temp[column] = temp[column].astype('category')
        
        if label_encoding == True:
            temp[column] = temp[column].cat.codes
            
    return temp
#Categorize nominal features:
continuous_columns = ['age','wage_per_hour','capital_gains','capital_losses',
                      'dividends_from_stocks','num_persons_worked_for_employer',
                      'instance_weight','weeks_worked_in_year']

nominal_columns = list(set(df.columns) - set(continuous_columns))
df_categorized = categorize_columns(nominal_columns)
###############################################################################
# Find values with ? in it:
for i in df.columns:
    t = df[i].value_counts()
    index = list(t.index)
    print ("The Value Counts of ? in", i)
    for i in index:
        temp = 0
        if i == ' ?':
            print (t[' ?'])
            temp = 1
            break
    if temp == 0:
        print ("0")
###############################################################################
# This distribution plot shows the distribution of Age of people across the Data Set
"plt.rcParams['figure.figsize'] = [8, 6]
sns.set_style("white")
fontsize= 16

sns.distplot(df['age'], bins = 45, color = '#2ab1ac')
plt.ylabel("Distribution", fontsize = fontsize)
plt.xlabel("Age", fontsize = fontsize)
plt.title('Distribution of Age', fontsize=fontsize + 4)
plt.margins(x = 0)

print ("The maximum age is", df['age'].max())
print ("The minimum age is", df['age'].min())"
###############################################################################
#Try a RFC
from sklearn.ensemble import RandomForestClassifier

X_train = categorize_columns(nominal_columns, label_encoding=True)
y_train= X_train.pop('y')

X_test = categorize_columns(nominal_columns, df =df_test, label_encoding = True)
y_test = X_test.pop('y')

model = RandomForestClassifier(class_weight='balanced')

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

model.score(X_test,y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


#Check if there are any NaN values
df1 = df_test[df_test.isna().any(axis=1)]
df.isnull().sum(axis = 0)

# Find values with ? in it:
for i in df.columns:
    t = df[i].value_counts()
    index = list(t.index)
    print ("The Value Counts of ? in", i)
    for i in index:
        temp = 0
        if i == ' ?':
            print (t[' ?'])
            temp = 1
            break
    if temp == 0:
        print ("0")
        
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(20,16))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

plot_feature_importance(model.feature_importances_,X_train.columns,'RANDOM FOREST')

###############################################################################
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# predict the results
y_pred=clf.predict(X_test)

# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

