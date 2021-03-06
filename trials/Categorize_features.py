# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:55:15 2021

@author: Ege
"""
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

X_train,X_test = categorize_columns(nominal_columns)
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


X_train,X_test = categorize_columns(nominal_columns)

X_train = categorize_education(X_train)
X_train = categorize_detailed_household(X_train,simple=False)
y_train= X_train.pop('y')

X_test = categorize_education(X_test)
X_test = categorize_detailed_household(X_test, simple = False)
y_test= X_test.pop('y')
###############################################################################
#Create hasCapitalGains and hasCapitalLosses, hasDividends
df['capital_gains'] > 0

#!!!!!!Ayr?? ayr?? bunlar?? categorize edemem train ile testi ????nk?? ayn?? label'?? verip vermedi??i belli de??il.
#Map Age as education
###############################################################################

def categorize_detailed_household(df, simple= False):
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
    di2_simple={
        ' Householder': 1,
        ' Spouse of householder': 1,
        
        ' Child <18 never marr RP of subfamily': 2,
        ' Child <18 never marr not in subfamily': 2,
        ' Child <18 ever marr RP of subfamily': 2,
        ' Child <18 spouse of subfamily RP': 2,
        ' Child <18 ever marr not in subfamily': 2,
        
        ' Child 18+ ever marr RP of subfamily': 3,
        ' Child 18+ never marr Not in a subfamily': 3,
        ' Child 18+ never marr RP of subfamily' : 3,
        ' Child 18+ spouse of subfamily RP': 3,
        ' Child 18+ ever marr Not in a subfamily':3,
        
        ' Grandchild <18 never marr RP of subfamily': 4,
        ' Grandchild <18 never marr child of subfamily RP':4,
        ' Grandchild <18 never marr not in subfamily': 4,
        ' Grandchild <18 ever marr RP of subfamily': 4,
        ' Grandchild <18 ever marr not in subfamily': 4,
        ' Grandchild 18+ never marr RP of subfamily': 4,
        ' Grandchild 18+ never marr not in subfamily': 4,   
        ' Grandchild 18+ ever marr RP of subfamily': 4,   
        ' Grandchild 18+ spouse of subfamily RP': 4,
        ' Grandchild 18+ ever marr not in subfamily':4,
        
        ' Other Rel <18 never married RP of subfamily': 5,
        ' Other Rel <18 never marr child of subfamily RP': 5, 
        ' Other Rel <18 never marr not in subfamily': 5,
        ' Other Rel <18 ever marr RP of subfamily': 5,
        ' Other Rel <18 spouse of subfamily RP': 5,
        ' Other Rel <18 ever marr not in subfamily': 5,
        ' Other Rel 18+ never marr RP of subfamily': 5,
        ' Other Rel 18+ never marr not in subfamily': 5,
        ' Other Rel 18+ ever marr RP of subfamily': 5,
        ' Other Rel 18+ spouse of subfamily RP': 5,
        ' Other Rel 18+ ever marr not in subfamily': 5,
        ' RP of unrelated subfamily': 5,
        ' Spouse of RP of unrelated subfamily': 5,
        ' Child under 18 of RP of unrel subfamily': 5,
        
        ' Nonfamily householder': 6,
        ' Secondary individual': 6,
        ' In group quarters': 6,
        
        }
    if simple == True:
        temp['detailed_household_and_family_stat'] = temp['detailed_household_and_family_stat'].map(di2_simple).fillna(df['detailed_household_and_family_stat'])
    else:
        temp['detailed_household_and_family_stat'] = temp['detailed_household_and_family_stat'].map(di2).fillna(df['detailed_household_and_family_stat'])
    return temp

X_train = categorize_columns(df,nominal_columns, label_encoding=True)
X_train = categorize_education(X_train)
X_train = categorize_detailed_household(X_train,simple=False)
y_train= X_train.pop('y')


X_test = categorize_columns(df_test,nominal_columns,label_encoding=True)
X_test = categorize_education(X_test)
X_test = categorize_detailed_household(X_test, simple = False)
y_test= X_test.pop('y')

###############################################################################
Imbalance ?? tekrar dene

Correlation matrixe bak y ile korrelasyon edenleri pdften ??ek categorization??n?? yap.
Label encodingini d??zelt.
agei categorilize 42.sayfada pdfte

continuous_columns = ['age','wage_per_hour','capital_gains','capital_losses',
                      'dividends_from_stocks','num_persons_worked_for_employer',
                      'instance_weight','weeks_worked_in_year']
nominal_columns = list(set(df.columns) - set(continuous_columns))

X_train,X_test = categorize_columns(nominal_columns, train=df, test=df_test)
y_train= X_train.pop('y')
y_test= X_test.pop('y')


model = RandomForestClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)

print(abs(y_test.value_counts()[0] - y_pred.value_counts()[0]), 'wrong predictions')