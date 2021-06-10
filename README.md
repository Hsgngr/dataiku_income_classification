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
- [Results](#results)

## Getting Started

This repository aims to explain the process of how I handled the project. Since the task has also a presentation part, I will keep this repository as technical as possible. Then I will filter my findings out and create more representable insights for my non-technical audience.

* `clean_main.py` is the main python script.
* `feature_engineering_clean.py` is the class that I created for preprocessing the data. It is imported in the `clean_main.py`. 
* Presentation can be found as `"Task Presentation.pptx"`
* `eda.py` and `eda_2.py` are for exploratory data analysis (they can be a little messy). 
* `feature_selector` is a generic library that I use in classification problems such as this.
* Other experiments are in the folder `/trials`




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
#### Let's compare training and test sets:

1) When we plot the distribution of Census data by age, in both training and test data look very similar. This makes our task easier since the training and test domain will be same.

![image](https://user-images.githubusercontent.com/32769732/120935774-35df2d00-c6b9-11eb-90b1-fd4b84c3bc6a.png)

2) However there is an imbalance in between the datasets for the labels. Lets plot them:

![image](https://user-images.githubusercontent.com/32769732/120935802-5b6c3680-c6b9-11eb-85d5-97bceef5dc7f.png)

As preprocessing we should handle this imbalance data. Let's keep exploring the data and features:

3) Here is the distribution of Age by Income
![image](https://user-images.githubusercontent.com/32769732/120935850-90788900-c6b9-11eb-91da-bd1f2238ed76.png)
* On the above graph we can see the how less many people have income more than 50K compare to others as the graph shows the total count of people.
* On the below graph, we see the density of both groups:
    * People from all age can have less income that 50K but only people who are older than 20 have more income than 50K. People with less income than 50K are slightly decreasing with the age.
    * People have more income than 50K has a normal distribution with a peak at 50.
4) Distribution of wealth between genders is as follows:
![image](https://user-images.githubusercontent.com/32769732/120940768-c296e480-c6d3-11eb-9099-cd0b4535ffd9.png)

* In the dataset the ratio of men and women are not equal, there is a small difference. However the most significant thing is the wealthier men's percentage is 5 times more than wealthier women.


## Data Preprocessing and Feature Engineering
I used my own library to extract feature importances for the task:
![image](https://user-images.githubusercontent.com/32769732/120941836-e3623880-c6d9-11eb-9047-18158b1e0c64.png)
These are the most important classes - subclasses. Instance weight is an interesting one because I'dont know anything about that task so first I will learn about that. Also let's plot the correlation matrix:
![image](https://user-images.githubusercontent.com/32769732/120942495-2d4d1d80-c6de-11eb-8d30-644df2af7212.png)


## Data Modelling
![image](https://user-images.githubusercontent.com/32769732/121442435-89af7780-c940-11eb-8103-cf302996d67a.png)

## Results

Steps:
* preprocessing (categorizing the data)
* feature engineering ( generating new features)
* feature selection
* ensemble learning (stacking)
* model_evaluation and going back to other steps iteratively.

![image](https://user-images.githubusercontent.com/32769732/121442442-8e742b80-c940-11eb-9089-640922d076cc.png)
![image](https://user-images.githubusercontent.com/32769732/121442408-7d2b1f00-c940-11eb-8684-32836d2f8051.png)

### Conclusion
These are the 22 features lastly selected for the model.

* Age
* Class of Worker
* Industry Code
* Occupation Code
* Education
* Wage per hour
* Marital Status 
* Major Industry Code
* Major Occupation Code
* Sex
* Capital Gains
* Capital Losses
* Dividends from Stocks
* Tax Filer Status
* Detailed Household and family Statistics
* Detailed Household summary in household
* Instance weight
* Number of persons worked for employer
* Country of Birth Father
* Country of Birth Mother
* Own Business or Self Employed
* Weeks Worked In Years

The US Census Data shows that having more than $50K income is only valid for %7 US population.
Most of the individuals who have more income than $50K are white but as a ratio Asian or Pacific Islanders have higher probability.
Having Capital Gain, Losses, or dividend from a stock indicates the probability of more income than $50K
Distribution shows an inequality by gender, men gets %5 percent while women gets only %1.
Age is an indicator for predict the individuals with more money. There is a normal distribution on that while age doesnt have much indication for income less than $50K.
Marrried Civillians has more chance to earn more than $50K than never married people.

My model's accuracy of predicting people with less income is %98. The accuracy of prediction for people who has higher income is %60, with together the model's macro average is %79. The baseline was %46. There is always room for improvement for example data needs more features to predict the people who has more income, do they have real-estate, their living location etc.





