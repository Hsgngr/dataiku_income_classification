# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 01:07:42 2021

@author: Ege
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# This shows the hours per week according to the education of the person
#sns.set(rc={'figure.figsize':(12,8)})
X_train = categorize_education(df)
sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = X_train['education'], y = X_train['wage_per_hour'], data = df, ax=ax)
ax.set_xticklabels(list(di.keys()), rotation=90)
################################################################################################
X_train['y'] = X_train['y'].astype('category').cat.codes

sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = X_train['education'], y = X_train['y'], data = X_train, ax=ax)
ax.set_xticklabels(list(di.keys()), rotation=90)
################################################################################################
X_train = hasCapital(X_train)
# Numbers of pairs of bars you want
N = 2

# Data on X-axis
cg = X_train[X_train['hasCapitalGains'] ==0].y.value_counts()
cg_2 = X_train[X_train['hasCapitalGains'] ==1].y.value_counts()


# Specify the values of blue bars (height)
blue_bar = (cg[0]/(cg[0] + cg[1]),cg_2[0]/(cg_2[0] + cg_2[1]))
# Specify the values of orange bars (height)
orange_bar = (cg[1]/(cg[0] + cg[1]),cg_2[1]/(cg_2[0] + cg_2[1]))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind + width, orange_bar, width, label='Yes', color ='#FFA156')
plt.bar(ind, blue_bar , width, label='No', color= '#609BC7')


plt.ylabel('Percent')
plt.title('Does Person has Capital Gains ?', fontsize= 16)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Income < 50K','Income>50K'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
###############################################################################
X_train = hasCapital(X_train)
# Numbers of pairs of bars you want
N = 2

# Data on X-axis
cg = X_train[X_train['hasCapitalLosses'] ==0].y.value_counts()
cg_2 = X_train[X_train['hasCapitalLosses'] ==1].y.value_counts()


# Specify the values of blue bars (height)
blue_bar = (cg[0]/(cg[0] + cg[1]),cg_2[0]/(cg_2[0] + cg_2[1]))
# Specify the values of orange bars (height)
orange_bar = (cg[1]/(cg[0] + cg[1]),cg_2[1]/(cg_2[0] + cg_2[1]))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind + width, orange_bar, width, label='Yes', color ='#FFA156')
plt.bar(ind, blue_bar , width, label='No', color= '#609BC7')


plt.ylabel('Percent')
plt.title('Does Person has Capital Losses ?', fontsize= 16)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Income < 50K','Income>50K'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
###############################################################################
X_train = hasCapital(X_train)
# Numbers of pairs of bars you want
N = 2

# Data on X-axis
cg = X_train[X_train['hasStock'] ==0].y.value_counts()
cg_2 = X_train[X_train['hasStock'] ==1].y.value_counts()


# Specify the values of blue bars (height)
blue_bar = (cg[0]/(cg[0] + cg[1]),cg_2[0]/(cg_2[0] + cg_2[1]))
# Specify the values of orange bars (height)
orange_bar = (cg[1]/(cg[0] + cg[1]),cg_2[1]/(cg_2[0] + cg_2[1]))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind + width, orange_bar, width, label='Yes', color ='#FFA156')
plt.bar(ind, blue_bar , width, label='No', color= '#609BC7')


plt.ylabel('Percent')
plt.title('Does Person has Stock ?', fontsize= 16)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Income < 50K','Income>50K'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
###############################################################################
#Marital Status VS Income
X_train['y'] = X_train['y'].astype('category').cat.codes

di2={
        ' Married-civilian spouse present' : 1,
        ' Married-A F spouse present': 2,
        ' Married-spouse absent': 3,
        ' Widowed': 4,
        ' Divorced': 5,
        ' Separated': 6,
        ' Never married':7,
        }

sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = X_train['marital_status'], y = X_train['y'], data = X_train, ax=ax)
ax.set_xticklabels(list(di2.keys()), rotation=90)

###############################################################################
#Marital Status VS Income
X_train['y'] = X_train['y'].astype('category').cat.codes

di2={
        ' Married-civilian spouse present' : 1,
        ' Married-A F spouse present': 2,
        ' Married-spouse absent': 3,
        ' Widowed': 4,
        ' Divorced': 5,
        ' Separated': 6,
        ' Never married':7,
        }

sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = X_train['race'], y = X_train['y'], data = X_train, ax=ax)
ax.set_xticklabels(list(di2.keys()), rotation=90)
###############################################################################
#Major Industry Code
X_train['y'] = X_train['y'].astype('category').cat.codes

di2={
        ' Married-civilian spouse present' : 1,
        ' Married-A F spouse present': 2,
        ' Married-spouse absent': 3,
        ' Widowed': 4,
        ' Divorced': 5,
        ' Separated': 6,
        ' Never married':7,
        }

sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = df_rich['race'], y = df_rich['y'], data = df_rich, ax=ax)

###############################################################################
fig, ax = plt.subplots(figsize = (10,12))

race = ['White','Black','Asian or Pacific Islander','Other','Amer Indian Aluet or Eskimo']

sns.histplot(df_rich['race'], stat= 'count', ax= ax)
ax.set_xticklabels( rotation=90)
ax.set_title()


df_rich['race'].value_counts()

###############################################################################
df['y'] = df['y'].astype('category').cat.codes
sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,12))
sns_grad = sns.barplot(x = df['race'], y = df['y'], data = df, ax=ax)
race = ['White','Asian or Pacific Islander', 'Amer Indian Aluet or Eskimo', 'Black', 'Other']
ax.set_xticklabels(race,rotation=90)
