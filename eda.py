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
plt.rcParams['figure.figsize'] = [8, 6]
sns.set_style("white")
fontsize= 16

sns.distplot(df['age'], bins = 30, color = '#609BC7')
plt.ylabel("Distribution", fontsize = fontsize)
plt.xlabel("Age", fontsize = fontsize)
plt.title('Distribution of Age', fontsize=fontsize + 4)
plt.margins(x = 0)

print ("The maximum age is", df['age'].max())
print ("The minimum age is", df['age'].min())
###############################################################################
#Try a RFC
from sklearn.ensemble import RandomForestClassifier

X_train = categorize_columns(nominal_columns, label_encoding=True)
y_train= X_train.pop('y')

X_test = categorize_columns(nominal_columns, df =df_test, label_encoding = True)
y_test = X_test.pop('y')

model = RandomForestClassifier(n_estimators=1000, class_weight={0:0.10, 1:0.90})

model.fit(X_train,y_train)
#y_pred = model.predict(X_test)

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
###############################################################################

# TensorFlow and tf.keras
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,y_train,validation_split=0.2, epochs=10)

###############################################################################
full_data = [df, df_test]

for data in full_data: 
    plt.hist(y_train)


train=y_train.value_counts()
test= y_test.value_counts()


labels= ['TRAIN','TEST']
zero= [train[0],test[0]]
ones = [train[1],test[1]]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

#TRAIN TEST PLOT

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, zero, width, label='TRAIN', color= ['#609BC7','#609BC7'])
rects2 = ax.bar(x + width/2, ones, width, label='TEST', color = ['#FFA156','#FFA156'])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of Examples', fontsize=16)
ax.set_xlabel('Datasets', fontsize=16)
ax.set_title('Distribution of Census Data', fontsize = 20)
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()

#AGE PLOT
plt.rcParams['figure.figsize'] = [20, 8]
plt.rc('font',size = 20)
fig, ax = plt.subplots(1,2)

sns.set(font_scale=1.2)
#sns.set_style("whitegrid")
sns.set_style('whitegrid',{'axes.grid':True})

sns.histplot(df['age'], bins = 15, color = '#609BC7', ax= ax[0],stat= 'count')
sns.histplot(df_test['age'], bins = 15, color = '#FFA156', ax= ax[1], stat = 'count')
ax[0].set_xlabel('Age (TRAIN)')
ax[1].set_xlabel('Age (TEST)')



plt.ylabel("Distribution", fontsize = fontsize)
plt.xlabel("Age", fontsize = fontsize)
plt.title('Distribution of Age', fontsize=fontsize + 4)
plt.margins(x = 0)


#AGE PLOT
plt.rcParams['figure.figsize'] = [20, 8]
plt.rc('font',size = 20)
fig, ax = plt.subplots(1,2)
fig.suptitle('Distribution of Age', fontsize= 30)

sns.set(font_scale=1.5)
#sns.set_style("whitegrid")
sns.set_style('whitegrid',{'axes.grid':True})

sns.histplot(df['age'], bins = 15, color = '#609BC7', ax= ax[0],stat= 'density', label='TRAIN') 
sns.histplot(df_test['age'], bins = 15, color = '#FFA156', ax= ax[1], stat = 'density', label='TEST')
ax[0].set_xlabel('AGE')
ax[1].set_xlabel('AGE')
ax[0].legend()
ax[1].legend()


#LABEL PLOT
labels= ['TRAIN','TEST']
zero= [train[0],test[0]]
ones = [train[1],test[1]]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, zero, width, label=' Income < 50K', color= ['#609BC7','#609BC7'])
rects2 = ax.bar(x + width/2, ones, width, label='Income > 50K', color = ['#FFA156','#FFA156'])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of Examples')
ax.set_xlabel('Datasets')
ax.set_title('Distribution of Income between Datasets', fontsize = 30)
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()

###############################################################################
df['y'] = df['y'].astype('category').cat.codes
df_less = df[df['y'] == 0]
df_more = df[df['y'] == 1]

#AGE PLOT
plt.rcParams['figure.figsize'] = [20, 8]
plt.rc('font',size = 20)
fig, ax = plt.subplots(2,1)
fig.suptitle('Distribution of Age by Income', fontsize= 30)

sns.set(font_scale=1.5)
#sns.set_style("whitegrid")
sns.set_style('whitegrid',{'axes.grid':True})

sns.histplot(df_less['age'], bins = 30, color = '#609BC7', ax= ax[0],stat= 'count', label='Less than 50K') 
sns.histplot(df_more['age'], bins = 30, color = '#FFA156', ax= ax[0], stat = 'count', label='More than 50K')
sns.histplot(df_less['age'], bins = 30, color = '#609BC7', ax= ax[1],stat= 'density', label='Less than 50K') 
sns.histplot(df_more['age'], bins = 30, color = '#FFA156', ax= ax[1], stat = 'density', label='More than 50K')


ax[0].set_xlabel('')
ax[1].set_xlabel('AGE')
ax[0].legend()

###############################################################################
df['y'] = df['y'].astype('category').cat.codes
df_less = df[df['y'] == 0]
df_more = df[df['y'] == 1]

#AGE PLOT
plt.rcParams['figure.figsize'] = [20, 8]
plt.rc('font',size = 20)
fig, ax = plt.subplots(2,1)
fig.suptitle('Age Distribution with Income', fontsize= 30)

sns.set(font_scale=1.5)
#sns.set_style("whitegrid")
sns.set_style('whitegrid',{'axes.grid':True})

sns.histplot(df_less['age'], bins = 30, color = '#609BC7', ax= ax[0],stat= 'count', label='Less than 50K') 
sns.histplot(df_more['age'], bins = 30, color = '#FFA156', ax= ax[0], stat = 'count', label='More than 50K')
sns.histplot(df_less['age'], bins = 30, color = '#609BC7', ax= ax[1],stat= 'density', label='Less than 50K') 
sns.histplot(df_more['age'], bins = 30, color = '#FFA156', ax= ax[1], stat = 'density', label='More than 50K')


ax[0].set_xlabel('')
ax[1].set_xlabel('AGE')
ax[0].legend()
###############################################################################
#SEX PLOT
labels = ['Female','Male']
plt.rcParams['figure.figsize'] = [16,9]
plt.rc('font',size = 20)
fig, ax = plt.subplots()
fig.suptitle('Distribution of Income by Gender', fontsize= 30)

sns.set(font_scale=1.5)
# #sns.set_style("whitegrid")
# sns.set_style('whitegrid',{'axes.grid':True})

# plt.hist([df_less['sex'],df_more['sex']], histtype='bar', stacked= True, bins=2,
#          color = ['#609BC7','#FFA156'],
#          label = ['less than 50K','More than 50K'])

# plt.legend()

females_less = df_less['sex'].value_counts()[0]
male_less =df_less['sex'].value_counts()[1]
female_more = df_more['sex'].value_counts()[1]
male_more = df_more['sex'].value_counts()[0]

sizes = [females_less,male_less,female_more,male_more]
labels = ['Female (<50K)','Male (<50K)','Female (>50K)','Male (>50K)']
plt.rcParams['text.color'] = 'black'

fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, autopct='%1.1f%%', labeldistance=1, rotatelabels= True,
        startangle=90)
ax1.axis('equal') 
ax1.legend()
plt.tight_layout()
#sns.histplot(df_less['sex'], bins = 30, color = '#609BC7', ax= ax[1],stat= 'density', label='Less than 50K') 
#sns.histplot(df_more['sex'], bins = 30, color = '#FFA156', ax= ax[1], stat = 'density', label='More than 50K')


ax[0].set_xlabel('')
ax[1].set_xlabel('Sex')
ax[0].legend()
################################################################################
#AGE PLOT
plt.scatter(df['instance_weight'],df['y'])

df_sampled = resample(df,replace=False,n_samples=100000)

fig = px.scatter_3d(df_sampled, x='capital_gains', y='age', z='y',
                    color='y')
fig.show()

plt.rcParams['figure.figsize'] = [1,1]
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
