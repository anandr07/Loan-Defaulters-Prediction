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

DATA SET unique value count                           

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/913626ac-851c-48e5-9ce0-70d4deb10fb9)

NULL VALUES

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/3be0a8af-e6cb-46c3-96ad-12d8f9d1a16d)



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

## Deployment Link:
http://customer-default-prediction.herokuapp.com/ 

(The link maybe expired)

![image](https://github.com/anandr07/Loan-Defaulters-Prediction/assets/66896800/bbe51cb9-b4f5-46eb-972d-5c77a09841c1)




