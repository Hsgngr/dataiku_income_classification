# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:23:44 2021

@author: Ege
"""
import pandas as pd
import numpy as np

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

df = census_data.rename(columns = column_names_dict)
df_test = census_data_test.rename(columns = column_names_dict)
###############################################################################
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
continuous_columns = ['age','wage_per_hour','capital_gains','capital_losses',
                      'dividends_from_stocks','num_persons_worked_for_employer',
                      'instance_weight','weeks_worked_in_year',
                      'education','detailed_household_and_family_stat']
nominal_columns = list(set(df.columns) - set(continuous_columns))

def categorize_columns(columns,train=df, test=df_test):
    le = preprocessing.LabelEncoder()
    temp_train = train.copy()
    temp_test = test.copy()
    
    for feat in columns:
        temp_train[feat] = le.fit_transform(temp_train[feat])
        temp_test[feat] = le.transform(temp_test[feat])
    return temp_train, temp_test

X_train,X_test = categorize_columns(nominal_columns, train=df, test=df_test)
###############################################################################
def categorize_education(df):
    temp= df.copy()
    di={
        ' Children': 0,
        ' Less than 1st grade': 1,
        ' 1st 2nd 3rd or 4th grade': 2,
        ' 5th or 6th grade':3,
        ' 7th and 8th grade':4,
        ' 9th grade': 5,
        ' 10th grade': 6,
        ' 11th grade': 7,
        ' 12th grade no diploma': 8,
        ' High school graduate' : 9,
        ' Some college but no degree': 10,
        ' Associates degree-occup /vocational': 11,
        ' Associates degree-academic program': 12,
        ' Bachelors degree(BA AB BS)':13,
        ' Masters degree(MA MS MEng MEd MSW MBA)': 14,
        ' Prof school degree (MD DDS DVM LLB JD)': 15,
        ' Doctorate degree(PhD EdD)': 16,    
        }
    temp['education'] = temp['education'].map(di).fillna(df['education'])
    return temp

X_train = categorize_education(X_train)
X_test = categorize_education(X_test)
###############################################################################
def categorize_detailed_household(df):
    temp= df.copy()
    di2={
        ' Householder': 1,
        ' Spouse of householder': 2,
        ' Child <18 never marr RP of subfamily': 3,
        ' Child <18 never marr not in subfamily': 4,
        ' Child <18 ever marr RP of subfamily': 5,
        ' Child <18 spouse of subfamily RP': 6,
        ' Child <18 ever marr not in subfamily': 7,
        
        ' Child 18+ ever marr RP of subfamily': 8,
        ' Child 18+ never marr Not in a subfamily': 9,
        ' Child 18+ never marr RP of subfamily' : 10,
        ' Child 18+ spouse of subfamily RP': 11,
        ' Child 18+ ever marr Not in a subfamily': 12,
        
        ' Grandchild <18 never marr RP of subfamily': 23,
        ' Grandchild <18 never marr child of subfamily RP':24,
        ' Grandchild <18 never marr not in subfamily': 25,
        ' Grandchild <18 ever marr RP of subfamily': 26,
        ' Grandchild <18 ever marr not in subfamily': 29,
        ' Grandchild 18+ never marr RP of subfamily': 30,
        ' Grandchild 18+ never marr not in subfamily': 31,   
        ' Grandchild 18+ ever marr RP of subfamily': 32,   
        ' Grandchild 18+ spouse of subfamily RP': 33,
        ' Grandchild 18+ ever marr not in subfamily': 34,
        
        ' Other Rel <18 never married RP of subfamily': 35,
        ' Other Rel <18 never marr child of subfamily RP': 36, 
        ' Other Rel <18 never marr not in subfamily': 37,
        ' Other Rel <18 ever marr RP of subfamily': 38,
        ' Other Rel <18 spouse of subfamily RP': 39,
        ' Other Rel <18 ever marr not in subfamily': 40,
        ' Other Rel 18+ never marr RP of subfamily': 41,
        ' Other Rel 18+ never marr not in subfamily': 42,
        ' Other Rel 18+ ever marr RP of subfamily': 43,
        ' Other Rel 18+ spouse of subfamily RP': 44,
        ' Other Rel 18+ ever marr not in subfamily': 45,
        ' RP of unrelated subfamily': 46,
        ' Spouse of RP of unrelated subfamily': 47,
        ' Child under 18 of RP of unrel subfamily': 48,
        ' Nonfamily householder': 49,
        ' Secondary individual': 50,
        ' In group quarters': 51,
        
        }
    temp['detailed_household_and_family_stat'] = temp['detailed_household_and_family_stat'].map(di2).fillna(df['detailed_household_and_family_stat'])
    return temp

X_train =  categorize_detailed_household(X_train)
X_test = categorize_detailed_household(X_test)
###############################################################################


###############################################################################
y_train = X_train.pop('y')
y_test =  X_test.pop('y')

model = RandomForestClassifier()
model.fit(X_train,y_train)
print('Score in test data:',model.score(X_test,y_test))
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)

print(abs(y_test.value_counts()[0] - y_pred.value_counts()[0]), 'wrong predictions')