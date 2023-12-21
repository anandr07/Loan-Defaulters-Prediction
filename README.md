# Bank Loan Defaulters Prediction

## Business Problem:
The objective of the analysis is to predict whether the customer will fall under loan default or not. 

Summary:
A Bank accepts deposits from customers and from the corpus thus available, it lends to Borrowers who want to carry out certain Business activities for Growth and Profit. It is often seen that due to some reasons like failure of Business, the company making losses or the company becoming delinquent/bankrupt the loans are either not Paid in full or are Charged-off or are written off. The Bank is thus faced with the problem of identifying those Borrowers who can pay up in full and not lending to borrowers who are likely to default.
This model allows the bank to make an analysis based on relevant data and then decide to give the loan or not. Thus, the prediction model reduces the bank's risk of defaulting om a loan. 



## Project Architecture:
![Project_Architecture](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/94ca0866-c019-408e-9b2f-fd6b344cc65d)
![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/5baa77ba-ac74-4f17-a3e4-1c0629e584e7)


## Data Cleaning
1.KNN imputation
2.Median imputation

## Explanatory Data Analysis (EDA)
As per the graph for Disbursement Gross,more cases have less amount as the disbursement gross amount increases, chances of defaulting decreases.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/32092020-520b-491c-a782-f4bc774c9a7a)


Existing businesses have a marginally more chance to default than new businesses.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/dfd3cb12-8bb6-4cee-9cdd-bbe896866355)


5260 businesses have franchises and defaulting chances are less for businesses with franchises.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/f9bc27b1-7e11-4b19-9a7b-378cca65c05c)


If no jobs retained, defaulting is very less, then the chances of defaulting comes down as the jobs increases.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/81ea7310-f413-403f-a72d-6ed50b49f5a9)


Urban business more likely to default than rural businesses

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/c5e80ad8-1627-49cb-abf6-08c056c2c404)


If covered under LowDoc, then very unlikely to default.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/f1b34e00-01bd-4202-a16a-61c171cd4310)


Loans for 0-5 and 30-40 month term has more chance of defaulting, and 5-30 month term less chance of defaulting.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/ad457487-d280-4df2-ba23-58c662c03fc7)


As the number of employees in the business increase, chances of defaulting decreases.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/d647010f-4819-462c-a4db-01df20d2e444)


Chances of defaulting is least when jobs created is between 10 and 400, highest when greater than 400.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/a7db2bec-1f15-4b67-a9d1-b815bea4c13e)


For revolving line of credit the chances of default is less than non revolving line

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/c2c7e78c-7626-47a5-bee1-8a9e4cb18bd6)


when comparing with SBA_Appv, GrAppv and DisbursementGross variables, Disbursement gross variable is more informative. Why because, at the time of a customer reaches to the bank for loan request, the bank have no information about the SBA_Appv. So bank is considering that only the requested amount of that customer for risk prediction.

So dropping GrAppv and SBA_Appv from data for solving multicollinearity problem and it also fixed the duplication problem.

Dropping chargeoffprin column as this amount will not present when a customer ask for a loan to the bank


## Correlation Matrix 
Heatmap of correlation matrix after removing the GrAppv and SBA_apprv and chargeoffprin: multi collenearity problem solved.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/1971c740-0c66-4762-a954-f93977eaa551)


## Pairplot
For finding overlapping of dependent variable and relation between variables.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/c5db6585-a482-481e-8215-d1629946d0d3)


## Summary statistics of numerical Variables
Measuring Distribution of the data.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/4bd6328b-6267-4a4b-8cd7-03bc368c847b)

Variables are Positively skewed .
In Variable CreateJob, 75% of the data is lies between zero.  So we have to check the importance of that variable with the output.
RetainedJob variable is highly affected by outlier problem. Because, the 74% is 4 and maximum is 9500 is too large.


## Box Plot
For visualisation of outliers

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/942c0ff3-7f83-4025-bb2a-c459d0b8a555)


Based on the box plot, CreateJob and RetainedJob are influenced by outliers compare with  NoEmp variable.

Considering that, here we have no use of these three variables, so total number of Employees have high priority for model prediction. 


## Model Building
Variables Taken for Model Building:
1.DisbursementGross
2.Term
3.NoEmp
4.NewExist
5.FranchiseCode
6.UrbanRural
7.LowDoc
8.RevLineCr
9.MIS_Status (Output Variable)


## Count Plot of Output Variable
Visualisation of Imbalance of Train Data.

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/acf78d81-550f-4e24-83db-4d288e6799f7)

After train-test ,the output data is imbalance in train data.So, before model building we have to treat the imbalance of the train data set. Using over Sampling technique called SMOTE.


## Models Deployed
The distance measuring machine learning algorithms need standardization because of the independent variables used are in different scale. So, we fix the scaling issue with standardization technique. 

The below models have been completed after standardization :
1.Logistic Regression
2.KNN
3.SVM
4.Naive Bayes

Models completed without Standardization:
1.Decision Tree
2.Random Forest
3.XGB


## Results:
Highest Acccuracy achieved using XGB: 93.68%

Selected Model: XGB classifier with parameter tuning


XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, 
	colsample_bytree=1, gamma=0, gpu_id=-1, importance_type='gain', interaction_constraints='', 
	learning_rate=0.300000012, max_delta_step=0, max_depth=6, min_child_weight=5, missing=nan, 	monotone_constraints='()', n_estimators=109, n_jobs=0, num_parallel_tree=1, 
	random_state=0, reg_alpha=0, reg_lambda=1,scale_pos_weight=2,subsample=1,
	tree_method='exact', validate_parameters=1, verbosity=None) 

## Deployment Link:
http://customer-default-prediction.herokuapp.com/ 

(The link maybe expired)

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/bbe51cb9-b4f5-46eb-972d-5c77a09841c1)




