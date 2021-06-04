# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:39:59 2021

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
    24: 'instance_weightt',
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

#Check if there are any NaN values
df1 = df_test[df_test.isna().any(axis=1)]
