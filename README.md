# Dataiku Income Classification Task
The Dataiku Data Scientist Technical Assesment: Binary Classication Task 

This repository outlines my approach to the Dataiku's Data Science Task. To goal of the task is to classify ~300,000 individuals income whether it is below or above 50K $ per year by analyzing the provided data from the United States Census Bureau.

## Table of contents
- [Dataiku Income Classification Task](#dataiku-income-classification-task)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Data Modelling](#data-modelling)
- [Model Assessment](#model-assessment)
- [Results](#results)

## Getting Started

This repository aims to explain the process of how I handled the project. Since the task has also a presentation part, I will keep this repository as technical as possible. Then I will filter my findings out and create more representable insights for my non-technical audience.



## Exploratory Data Analysis

The first thing I noticed in the given Census data some of the columns are missing such as 'total_earnings' or 'adjusted_gross_income'. These are missing on purpose for creating the task. Here is the full list of which columns are missing:
```
adjusted gross income
federal income tax liability
**instance weight**
total person earnings
total person income
taxable income amount

**Note: Instance weight is not shown on the information.txt but it is in the data as the 24th column.
```

The train and test data was split in approximately 2/3, 1/3 proportions. 
```              
                  TRAIN       TEST
- 50K :           187141      93576      
+ 50K :           12382       6186        
```
However there is a huge imbalance between the labels. Most of the people (93%) who contributed to this survey has lower income than 50K. Even if we predict all of the test data as - 50K we will reach %93 accuracy. Therefore accuracy is not a good measurement metric for this problem.


The other first thing was understanding what is **Not in Universe** means because most of the features have it. It implies that the person was not a part of the population to which the question was directed. 
 There are 42 features in total, 35 nominal, 8 continuous including the label column. Here is the full list of features:

```  
FEATURE NAME                                 FEATURE TYPE                                 DISTINCT VALUE COUNT  
age                                          int64                                        continuous 
class_of_work                                category                                     9
industry_code                                category                                     52
occupation_code                              category                                     47
education                                    category                                     17
wage_per_hour                                int64                                        continous
enrolled_in_edu_inst_last_wk                 category                                     3
marital_status                               category                                     7
major_industry_code                          category                                     24
major_occupation_code                        category                                     15
race                                         category                                     5
hispanic_origin                              category                                     10
sex                                          category                                     2
member_of_labor_union                        category                                     3
reason_for_unemployment                      category                                     6
full_or_part_time_employment_stat            category                                     8
capital_gains                                int64                                        continous
capital_losses                               int64                                        continous
dividends_from_stocks                        int64                                        continous
tax_filer_status                             category                                     6
region of previous residence                 category                                     6
state_of_previous_residence                  category                                     51 (708 Unknown Value)
detailed_household_and_family_stat           category                                     38
detailed_household_summary_in_household      category                                     8
instance_weight                              float64                                      continous
migration_code_change_in_msa                 category                                     10 (99696 Unknown Value)
migration_code_change_in_reg                 category                                     9  (99696 Unknown Value)
migration_code_move_within_reg               category                                     10 (99696 Unknown Value)
live_in_this_house_1_year_ago                category                                     3
migration_prev_res_in_sunbelt                category                                     4  (99696 Unknown Value)
num_persons_worked_for_employer              int64                                        continous
family_member_under_18                       category                                     5
country_of_birth_father                      category                                     43 (6713 Unknown Value)
country_of_birth_mother                      category                                     43 (6119 Unknown Value)
country_of_birth_self                        category                                     43 (3393 Unknown Value)
citizenship                                  category                                     5
own_business_or_self_employed                category                                     3
fill_inc_questionnaire_for_veterans_admin    category                                     3
veteran_benefits                             category                                     3
weeks_worked_in_years                        category                                     53
year                                         category                                     2
label                                        category                                     2
```  

Bir sürü graph
## Data Preprocessing and Feature Engineering

Bir sürü feature
## Data Modelling
buraya Model summary() nasıl yaptığın
## Model Assessment
buraya işte precision accuracy roc, f1 graphlerini koy.
## Results
buraya sonuç ne, feature importance (rfc çalışırsa)


