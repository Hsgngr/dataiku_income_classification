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
                      'instance_weight','weeks_worked_in_years',
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
onehot = preprocessing.OneHotEncoder()

def categorize_onehot_columns(columns, train=df, test=df_test):
    temp_train = train.copy()
    temp_test = test.copy()
    
    onehot = preprocessing.OneHotEncoder()      
    onehot.fit(temp_train[nominal_columns])
    
    X_train_onehot=onehot.transform(temp_train[nominal_columns])
    X_test_onehot=onehot.transform(temp_test[nominal_columns])
    
    pd.concat([X_train[continuous_columns],X_train_onehot])
    
    return X_train, X_test

X_train, X_test = categorize_onehot_columns(nominal_columns, train=df, test= df_test)
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
    temp['education'] = temp['education'].map(di)
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
    temp['detailed_household_and_family_stat'] = temp['detailed_household_and_family_stat'].map(di2)
    return temp

X_train =  categorize_detailed_household(X_train)
X_test = categorize_detailed_household(X_test)
###############################################################################
#CLASS OF WORKER
def categorize_class_of_work(df):
    temp= df.copy()
    di2={
        ' Not in universe' : 0,
        ' Private': 1,
        ' Federal government': 2,
        ' State government': 2,
        ' Local government': 2,
        ' Self-employed-incorporated': 3,
        ' Self-employed-not incorporated': 3,
        ' Without pay': 4,
        ' Never worked': 4
        }
    temp['class_of_work'] = temp['class_of_work'].map(di2)
    return temp
###############################################################################
#AMARITL Marital Status
def categorize_marital_status(df):
    temp= df.copy()
    di2={
        ' Married-civilian spouse present' : 1,
        ' Married-A F spouse present': 2,
        ' Married-spouse absent': 3,
        ' Widowed': 4,
        ' Divorced': 5,
        ' Separated': 6,
        ' Never married':7,
        }
    temp['marital_status'] = temp['marital_status'].map(di2)
    return temp
###############################################################################
#HasCapitalGain
def hasCapital(df):
    temp = df.copy()
    temp.loc[temp['capital_gains'] >0, 'hasCapitalGains']= 1
    temp.loc[temp['capital_gains'] == 0, 'hasCapitalGains']= 0
    
    temp.loc[temp['capital_losses'] >0, 'hasCapitalLosses']= 1
    temp.loc[temp['capital_losses'] == 0, 'hasCapitalLosses']= 0
    
    temp.loc[temp['dividends_from_stocks'] >0, 'hasStock']= 1
    temp.loc[temp['dividends_from_stocks'] == 0, 'hasStock']= 0
    
    return temp
###############################################################################
#Citizenship
def categorize_citizen(df):
    temp = df.copy()
    citizen_mapping = {
        " Native- Born in the United States": 1,
        ' Foreign born- Not a citizen of U S ': 5,
        " Foreign born- U S citizen by naturalization": 4,
        " Native- Born abroad of American Parent(s)": 3,
        " Native- Born in Puerto Rico or U S Outlying": 2
        }
    native_mapping = {
        " Native- Born in the United States": 1,
        ' Foreign born- Not a citizen of U S ': 2,
        " Foreign born- U S citizen by naturalization": 2,
        " Native- Born abroad of American Parent(s)": 1,
        " Native- Born in Puerto Rico or U S Outlying": 1
        }
    
    temp['isNative']= temp['citizenship'].map(native_mapping)
    temp['citizenship'] = temp['citizenship'].map(citizen_mapping)
    
    return temp
    
###############################################################################
# Detailed_household_summary_in_household

def categorize_detailed_household_summary(df):
    temp = df.copy()
    hdi = {
        " Householder": 1,
        ' Spouse of householder': 2,
        " Child under 18 never married": 3,
        " Child under 18 ever married": 4,
        " Child 18 or older": 5,
        " Other relative of householder" : 6,
        " Nonrelative of householder" : 7,
        " Group Quarters- Secondary individual" : 8,
        }

    
    temp['detailed_household_summary_in_household']= temp['detailed_household_summary_in_household'].map(hdi)
    
    return temp
###############################################################################  
#Normalize column
from sklearn import preprocessing
def normalize_column(column, train, test):
    temp_train= train.copy()
    temp_test = test.copy()
    
    train_values= temp_train[column].values.reshape(-1, 1)
    test_values = temp_test[column].values.reshape(-1, 1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    train_values_normalized = min_max_scaler.fit_transform(train_values)
    test_values_normalized = min_max_scaler.transform(test_values)
    
    
    temp_train[column] = train_values_normalized
    temp_test[column] = test_values_normalized
    return temp_train,temp_test  
  
###############################################################################
continuous_columns = ['age','wage_per_hour','capital_gains','capital_losses',
                      'dividends_from_stocks','num_persons_worked_for_employer',
                      'instance_weight','weeks_worked_in_years',
                      'education','detailed_household_and_family_stat','class_of_work',
                      'marital_status','citizenship','detailed_household_summary_in_household']
nominal_columns = list(set(df.columns) - set(continuous_columns))

def preprocess_data(df,df_test):
    X_train,X_test = categorize_columns(nominal_columns, train=df, test=df_test)
    
    X_train = categorize_education(X_train)
    X_test = categorize_education(X_test)
    
    X_train =  categorize_detailed_household(X_train)
    X_test = categorize_detailed_household(X_test)
    
    X_train = categorize_class_of_work(X_train)
    X_test = categorize_class_of_work(X_test)
    
    X_train = categorize_marital_status(X_train)
    X_test = categorize_marital_status(X_test)
    
    X_train = hasCapital(X_train)
    X_test = hasCapital(X_test)
    
    X_train = categorize_citizen(X_train)
    X_test = categorize_citizen(X_test)
    
    X_train =categorize_detailed_household_summary(X_train)
    X_test  =categorize_detailed_household_summary(X_test)
    
    X_train,X_test = normalize_column('instance_weight',X_train, X_test)
    
    return X_train, X_test
###############################################################################


from sklearn.ensemble import RandomForestClassifier

X_train,X_test = preprocess_data(df,df_test)
y_train = X_train.pop('y')
y_test =  X_test.pop('y')

model = RandomForestClassifier()
model.fit(X_train,y_train)
print('Score in test data:',model.score(X_test,y_test))
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
###############################################################################
import joblib

joblib.dump(X_train,'X_train.pkl')
joblib.dump(y_train,'y_train.pkl')

joblib.dump(X_test,'X_test.pkl')
joblib.dump(y_test,'y_test.pkl')
