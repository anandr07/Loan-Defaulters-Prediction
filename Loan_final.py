# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:54:01 2020

@author: priti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
import missingno as mn
import datetime
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import classification_report , roc_curve , auc
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle


Rawdata=pd.read_csv("C:\\Users\\priti\\Desktop\\Anand\\Project\\bank_final.csv")
Rawdata.head()
Rawdata.describe()
Rawdata.dtypes
Rawdata.shape
Rawdata.columns
Rawdata.isna().sum()
Rawdata.nunique()

#Visualise Na values in dataframe
mn.bar(Rawdata)
mn.heatmap(Rawdata)

#Categorical data (Checking zeros proportions and class imbalance)
(Rawdata['City'].value_counts()/Rawdata['City'].count())*100
(Rawdata['State'].value_counts()/Rawdata['State'].count())*100
(Rawdata['Bank'].value_counts()/Rawdata['Bank'].count())*100
(Rawdata['CCSC'].value_counts()/Rawdata['CCSC'].count())*100
(Rawdata['Term'].value_counts()/Rawdata['Term'].count())*100
(Rawdata['Bank'].value_counts()/Rawdata['Bank'].count())*100
(Rawdata['NewExist'].value_counts()/Rawdata['NewExist'].count())*100
(Rawdata['Term'].value_counts()/Rawdata['Term'].count())*100
(Rawdata['RevLineCr'].value_counts()/Rawdata['RevLineCr'].count())*100
(Rawdata['UrbanRural'].value_counts()/Rawdata['UrbanRural'].count())*100
(Rawdata['LowDoc'].value_counts()/Rawdata['LowDoc'].count())*100
(Rawdata['MIS_Status'].value_counts()/Rawdata['MIS_Status'].count())*100
(Rawdata['FranchiseCode'].value_counts()/Rawdata['FranchiseCode'].count())*100


#Continous data cleaning
Rawdata['DisbursementGross']=Rawdata.DisbursementGross.apply(lambda x:x.strip('$'))
Rawdata['BalanceGross']=Rawdata.BalanceGross.apply(lambda x:x.strip('$'))
Rawdata['ChgOffPrinGr']=Rawdata.ChgOffPrinGr.apply(lambda x:x.strip('$'))
Rawdata['SBA_Appv']=Rawdata.SBA_Appv.apply(lambda x:x.strip('$'))
Rawdata['GrAppv']=Rawdata.GrAppv.apply(lambda x:x.strip('$'))
Rawdata['DisbursementGross']=Rawdata.DisbursementGross.apply(lambda x:x.replace(',',''))
Rawdata['BalanceGross']=Rawdata.BalanceGross.apply(lambda x:x.replace(',',''))
Rawdata['ChgOffPrinGr']=Rawdata.ChgOffPrinGr.apply(lambda x:x.replace(',',''))
Rawdata['SBA_Appv']=Rawdata.SBA_Appv.apply(lambda x:x.replace(',',''))
Rawdata['GrAppv']=Rawdata.GrAppv.apply(lambda x:x.replace(',',''))
Rawdata['DisbursementGross']=Rawdata['DisbursementGross'].astype(float)
Rawdata['BalanceGross']=Rawdata['BalanceGross'].astype(float)
Rawdata['ChgOffPrinGr']=Rawdata['ChgOffPrinGr'].astype(float)
Rawdata['SBA_Appv']=Rawdata['SBA_Appv'].astype(float)
Rawdata['GrAppv']=Rawdata['GrAppv'].astype(float)
Rawdata.dtypes

(Rawdata.DisbursementGross==0).sum()
(Rawdata['DisbursementGross'].value_counts()/Rawdata['DisbursementGross'].count())*100
(Rawdata['BalanceGross'].value_counts()/Rawdata['BalanceGross'].count())*100
(Rawdata['ChgOffPrinGr'].value_counts()/Rawdata['ChgOffPrinGr'].count())*100
(Rawdata['SBA_Appv'].value_counts()/Rawdata['SBA_Appv'].count())*100


#Cleaning of data

#ChargeOffDate is having 73% of NA values so the variable is not taken in consideration
Newdata=Rawdata.drop(['ChgOffDate'],axis=1)
Newdata=pd.DataFrame(Newdata)
Newdata.nunique()

#Name,City,Bank,CCSC and Zip will be irrelavant to the response variable 
Newdata=Newdata.drop(['Name'],axis=1)
Newdata=Newdata.drop(['City'],axis=1)
Newdata=Newdata.drop(['Zip'],axis=1)
Newdata=Newdata.drop(['Bank'],axis=1)
Newdata=Newdata.drop(['CCSC'],axis=1)
#RevlineCr have a lot of garbage values which decreases the rows of Data significantly so is dropped
Newdata=Newdata.drop(['RevLineCr'],axis=1)

#State
Newdata['State'].value_counts()
Newdata[['State','MIS_Status']].groupby(['State']).mean().sort_values(by='MIS_Status',ascending=False)
#FL state has highest probabilty to default and VT least
sns.countplot(x='State',data=Newdata) # more loans from CA and NY then FL,TX, OH
plt.show()

#BankState
Newdata['BankState'].value_counts() #Most banks in NC least in PR
Newdata[['BankState','MIS_Status']].groupby(['BankState']).mean().sort_values(by='MIS_Status',ascending=False)# VA highest, MA least
#Bank in VA state has highest probabilty to default and MA least


#Also BankState and State have high unique values making difficult to intrepret 
Newdata=Newdata.drop(['BankState'],axis=1)
Newdata=Newdata.drop(['State'],axis=1)

#NA values (After dropping the above columns not much rows have NA values so can be dropped)
Newdata.isna().sum()
#Dropping Rows with NA values
Newdata = Newdata.dropna()
#Removing Noisy data 
Newdata.drop(Newdata[Newdata['LowDoc']=='1'].index,inplace=True)
Newdata.drop(Newdata[Newdata['LowDoc']=='C'].index,inplace=True)
Newdata=Newdata[(Newdata.FranchiseCode == 0) | (Newdata.FranchiseCode == 1)]

#Converting columns into category
Newdata['LowDoc'] = Newdata['LowDoc'].astype('category')
Newdata['MIS_Status'] = Newdata['MIS_Status'].astype('category')
Newdata['LowDoc'] = Newdata.LowDoc.cat.codes
Newdata['MIS_Status'] = Newdata.MIS_Status.cat.codes
#LowDoc(Y=1,N=0),MIS_Status(PIF=1,CHGOFF=0)

#Feature Generation
#Handling the date-time variables 
Newdata['ApprovalDate']= pd.to_datetime(Newdata['ApprovalDate']) 
Newdata['DisbursementDate']= pd.to_datetime(Newdata['DisbursementDate'])
Newdata.dtypes
Newdata['Disbursementyear'] = Newdata['DisbursementDate'].dt.year
Newdata['DaysforDibursement'] = Newdata['DisbursementDate'] - Newdata['ApprovalDate']
Newdata['DaysforDibursement'] = Newdata.apply(lambda row: row.DaysforDibursement.days, axis=1)
#Removing the Date-time variables ApprovalDate and DisbursementDate
Newdata=Newdata.drop(['ApprovalDate','DisbursementDate'],axis=1)

#Skewness and kurtosis

Newdata['Term'].skew()
Newdata['NoEmp'].skew()
Newdata['DisbursementGross'].skew()
Newdata['ChgOffPrinGr'].skew()
Newdata['GrAppv'].skew()
Newdata['SBA_Appv'].skew()
Newdata['DaysforDibursement'].skew()
Newdata['Disbursementyear'].skew()


Newdata['Term'].kurtosis()
Newdata['NoEmp'].kurtosis()
Newdata['DisbursementGross'].kurtosis()
Newdata['ChgOffPrinGr'].kurtosis()
Newdata['GrAppv'].kurtosis()
Newdata['SBA_Appv'].kurtosis()
Newdata['DaysforDibursement'].kurtosis()
Newdata['Disbursementyear'].kurtosis()


#Visualising data

#MIS Status
plt.rcParams.update({'figure.figsize':(12,8)})
plt.show()
sns.countplot(x='MIS_Status',data=Newdata)
plt.title("0 : Non-defaulter and 1: defaulter")
plt.show()

#NewExist
Newdata.drop(Newdata[Newdata.NewExist.isna()].index,inplace=True)
sns.countplot(x='NewExist',hue='MIS_Status',data=Newdata)
plt.title("MIS_Status Vs NewExist")
plt.show()
#Existing business more than new business in dataset

#Franchise Code
sns.countplot(x='MIS_Status',hue='FranchiseCode',data=Newdata)
plt.title("MIS_Status vs franchisescode")
plt.show()
#Only few have franchises

#UrbanRural
sns.countplot(x='MIS_Status',hue='UrbanRural',data=Newdata)
plt.title("MIS_Status Vs UrbanRural")
plt.show()
#More cases of urban; majority of unidentified is in non-default

#LowDoc
sns.countplot(x='MIS_Status',hue='LowDoc',data=Newdata)#majority businesses are not under LowDoc
plt.title('MIS_Status VS LowDoc Y=0 & N=1')
plt.show()

#ChgOffPrinGr
pd.crosstab(Newdata.MIS_Status,Newdata.ChgOffPrinGr==0)
plt.show()
# If not defaulter then very less chance to have chargeoff amount
# If a defaulter then there are very few cases where the amount is not chargedoff

#ApprovalFY
sns.countplot(x='ApprovalFY',data=Newdata)# more approvals in 1997-1998 and 2004-2007
plt.show()
Newdata['ApprovalFY'].value_counts() #highest no of approvals in 2006 least in 1962,65,66 
sns.countplot(x='MIS_Status',hue='ApprovalFY',data=Newdata)
plt.title("MIS_Status Vs ApprovalFY")
plt.show()
Newdata[['ApprovalFY','MIS_Status']].groupby(['ApprovalFY']).mean().sort_values(by='ApprovalFY',ascending=True)
# If loan is approved before 1982, high probability to default; 1997-2003 very less chance to default 

#Term  
sorted(Newdata['Term'].unique()) # min=0 , max=480
sns.distplot(Newdata['Term'])
plt.show()
sns.boxplot(x='MIS_Status',y='Term',data=Newdata)
plt.title("MIS_Status VS Term")
plt.show()

#NoEmp 
sorted(Newdata['NoEmp'].unique())# min=0 ; max=9999
sns.distplot(Newdata['NoEmp'])
sns.boxplot(x='MIS_Status',y='NoEmp',data=Newdata)
plt.title("MIS_Status VS Number of employees")
plt.show()


plt.hist(Newdata['ApprovalFY'])
plt.hist(Newdata['NewExist'])
plt.hist(Newdata['FranchiseCode'])
plt.hist(Newdata['UrbanRural'])
plt.hist(Newdata['LowDoc'])
plt.hist(Newdata['BalanceGross'])
plt.hist(Newdata['MIS_Status'])

sns.distplot(Newdata['Term'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['NoEmp'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['DisbursementGross'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['ChgOffPrinGr'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['GrAppv'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['SBA_Appv'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['DaysforDibursement'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})
sns.distplot(Newdata['Disbursementyear'],hist=True,kde=True,hist_kws={'edgecolor':'black'},color='darkblue',kde_kws={'linewidth': 2})

plt.scatter(Newdata['MIS_Status'],Newdata['Term'],c ="pink",linewidths = 2,marker ="s",edgecolor ="green",s = 50) 
plt.scatter(Newdata['MIS_Status'],Newdata['NoEmp'],c ="pink",linewidths = 2,marker ="s",edgecolor ="green",s = 50) 
plt.scatter(Newdata['MIS_Status'],Newdata['ChgOffPrinGr'],c ="pink",linewidths = 2,marker ="s",edgecolor ="green",s = 50) 
plt.scatter(Newdata['MIS_Status'],Newdata['DaysforDibursement'],c ="pink",linewidths = 2,marker ="s",edgecolor ="green",s = 50) 
plt.scatter(Newdata['MIS_Status'],Newdata['Disbursementyear'],c ="pink",linewidths = 2,marker ="s",edgecolor ="green",s = 50) 

sns.relplot(x='Term',y='SBA_Appv',hue='MIS_Status',data=Newdata)
sns.relplot(x='NoEmp',y='SBA_Appv',hue='MIS_Status',data=Newdata)
sns.relplot(x='ChgOffPrinGr',y='SBA_Appv',hue='MIS_Status',data=Newdata)
sns.relplot(x='DaysforDibursement',y='SBA_Appv',hue='MIS_Status',data=Newdata)

Correlation=Newdata.corr()
sns.heatmap(Correlation,xticklabels=Correlation.columns,yticklabels=Correlation.columns)
#From heatmap we can see that ApprovalFY is correlated to Disbursement year and SBA_Appv is correlated to Gr_Appv
#So we can drop columns ApprovalFY and GrAppv

Newdata=Newdata.drop(['ApprovalFY','GrAppv'],axis=1)

Correlation=Newdata.corr()
sns.heatmap(Correlation,xticklabels=Correlation.columns,yticklabels=Correlation.columns)
#Their is still correlation between DisbusmentGross and SBA_Appv
Correlation['SBA_Appv']

Newdata=Newdata.drop(['DisbursementGross'],axis=1)

Correlation=Newdata.corr()
sns.heatmap(Correlation,xticklabels=Correlation.columns,yticklabels=Correlation.columns)
Correlation['UrbanRural']
#Their is some correlation between Disbursementyear and UrbanRural
Newdata=Newdata.drop(['Disbursementyear'],axis=1)

#Calculating VIF of all variables
y, x = dmatrices('MIS_Status ~ DaysforDibursement+Term+NoEmp+BalanceGross+ChgOffPrinGr+SBA_Appv ',Newdata,return_type='dataframe')

vif = pd.DataFrame()
vif["Variables"] = x.columns
vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif
#No collinearity problem found

#FEATURE SELECTION

#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
#Positiong MIS_Status last
Newdata=Newdata[['Term','NoEmp','NewExist','CreateJob','RetainedJob','FranchiseCode','UrbanRural','LowDoc','BalanceGross','ChgOffPrinGr','SBA_Appv','DaysforDibursement','MIS_Status']]
X=Newdata.iloc[:,0:12]
Y=Newdata.iloc[:,-1]
Y=pd.DataFrame(Y)
Model_ETC=ExtraTreesClassifier(n_estimators=10)
Model_ETC.fit(X,Y)
print(Model_ETC.feature_importances_)
N=12
ind=np.arange(N)
plt.bar(range(len(Model_ETC.feature_importances_)),Model_ETC.feature_importances_)
plt.xticks(ind,('Term','NoEmp','NewExist','CreateJob','RetainedJob','FranchiseCode','UrbanRural','LowDoc','BalanceGross','ChgOffPrinGr','SBA_Appv','DaysforDibursement'),rotation='vertical')

#XGB for feature selection
from xgboost import XGBClassifier
Model_XGB=XGBClassifier()
Model_XGB.fit(X,Y)
print(Model_XGB.feature_importances_)
plt.bar(range(len(Model_XGB.feature_importances_)),Model_XGB.feature_importances_)
plt.xticks(ind,('Term','NoEmp','NewExist','CreateJob','RetainedJob','FranchiseCode','UrbanRural','LowDoc','BalanceGross','ChgOffPrinGr','SBA_Appv','DaysforDibursement'),rotation='vertical')

#ExtraTreeClassifier and XGBoost both show that BalanceGross is not a good feature.

Newdata=Newdata.drop(["BalanceGross"],axis=1)

X=Newdata.iloc[:,0:11]
Y=Newdata.iloc[:,-1]

#Using XGBoost
data_dmatrix = xgb.DMatrix(data=X,label=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
Model_XGB.fit(X_train,Y_train)
print(Model_XGB)
#Train pedictions
X_pred=Model_XGB.predict(X_train)
XGB_predictions_train = [round(value) for value in X_pred]
accuracy_XGB_train = accuracy_score(Y_train, XGB_predictions_train)
print("Accuracy: %.2f%%" % (accuracy_XGB_train * 100.0))
#Train accuracy=99.23%
#Test predictions
Y_pred=Model_XGB.predict(X_test)
XGB_predictions = [round(value) for value in Y_pred]
# Evaluate predictions
accuracy_XGB = accuracy_score(Y_test, XGB_predictions)
print("Accuracy: %.2f%%" % (accuracy_XGB * 100.0))
confusion_matrix(Y_test,XGB_predictions)
matrix = classification_report(Y_test,XGB_predictions)
print(matrix)
#Test accuracy_XGB=98.99%

#ROC of XGB model
fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(Y_test, XGB_predictions)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_xgb, tpr_xgb, 'b', label = 'AUC = %0.2f' % roc_auc_xgb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#XGBoost with CrossValidation
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1, 
                'max_depth': 5, 'alpha': 10}
CV_rmse = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print(CV_rmse.head())
print((CV_rmse["test-rmse-mean"]).tail(1))
xg_pred = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
#Predictions
dmatrix_xtest=xgb.DMatrix(X_test)
CV_pred=xg_pred.predict(dmatrix_xtest)
CV_predictions=[round(i) for i in CV_pred]
accuracy_CV = accuracy_score(Y_test, CV_predictions)
print("Accuracy: %.2f%%" % (accuracy_CV * 100.0))
#XGB_CV=98.43%


# Random forest algorithm for classification
Model_RF = RandomForestClassifier(n_estimators=100)
Model_RF.fit(X_train, Y_train)
RF_predictions= Model_RF.predict(X_train)
print("Accuracy:",accuracy_score(Y_train,RF_predictions)*100)
#Accuracy_RF_train=100%
RF_predictions_test=Model_RF.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,RF_predictions_test)*100)
#Accuracy_RF_test=98.98%
pd.crosstab(Y_test,RF_predictions_test)
print(confusion_matrix(Y_test,RF_predictions_test))
print(classification_report(Y_test, RF_predictions_test))
RF_CV_score = cross_val_score(Model_RF, X, Y, cv=10, scoring='accuracy')
print(RF_CV_score)
print("Accuracy:", np.mean(RF_CV_score)*100)
#AUC_RF_CV=98.86%

#ROC Curve of Random forest model
fpr_RF, tpr_RF, threshold_RF = roc_curve(Y_test, RF_predictions_test)
roc_auc_RF = auc(fpr_RF, tpr_RF)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_RF, tpr_RF, 'b', label = 'AUC = %0.2f' % roc_auc_RF)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Navie Bayes
Model_gnb = GaussianNB()
Model_gnb.fit(X_train, Y_train)
gnb_predictions= Model_gnb.predict(X_train)
print("Accuracy:",accuracy_score(Y_train,gnb_predictions)*100)
#Train_accuracy_GNB=97.72%
gnb_predictions_test= Model_gnb.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,gnb_predictions_test)*100)
#Test_accuracy_test=97.89%
CV_NB=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=100)
Score_NB=cross_val_score(Model_gnb,X,Y,scoring='accuracy',cv=CV_NB,n_jobs=-1,error_score='raise')
print('Accuracy: %.3f ' % (np.mean(Score_NB)*100))
#Accuracy_CV_gnb=97.54%

#ROC Curve of Random Naive Bayes
fpr_NB, tpr_NB, threshold_NB = roc_curve(Y_test, gnb_predictions_test)
roc_auc_NB = auc(fpr_NB, tpr_NB)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_NB, tpr_NB, 'b', label = 'AUC = %0.2f' % roc_auc_NB)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Light_GradientBoostingMethod
Model_LGBM=LGBMClassifier()
CV_LGBM=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=100)
Score_LGBM=cross_val_score(Model_LGBM,X,Y,scoring='accuracy',cv=CV_LGBM,n_jobs=-1,error_score='raise')
print('Accuracy: %.3f' % (np.mean(Score_LGBM)*100))
#Accuracy_LGBM_CV=99.02%
Model_LGBM.fit(X_train,Y_train)
LGBM_pred_train=Model_LGBM.predict(X_train)
print("Accuracy_train_LGBM:",accuracy_score(Y_train,LGBM_pred_train)*100)
#Accuracy_LGBM_train=99.14%
LGBM_pred_test=Model_LGBM.predict(X_test)
print("Accuracy_train_LGBM:",accuracy_score(Y_test,LGBM_pred_test)*100)
#ACCuracy_LGBM_test=98.99%

#ROC Curve of Light_GradientBoosting 
fpr_LGBM, tpr_LGBM, threshold_LGBM = roc_curve(Y_test, LGBM_pred_test)
roc_auc_LGBM = auc(fpr_LGBM, tpr_LGBM)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_LGBM, tpr_LGBM, 'b', label = 'AUC = %0.2f' % roc_auc_LGBM)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Table of all models
data_val=[['XGBoost',accuracy_XGB_train*100.0,accuracy_XGB*100.0],
          ['RandomForest',accuracy_score(Y_train,RF_predictions)*100,accuracy_score(Y_test,RF_predictions_test)*100],
          ['NaiveBayes',accuracy_score(Y_train,gnb_predictions)*100,accuracy_score(Y_test,gnb_predictions_test)*100],
          ['LightGB',accuracy_score(Y_train,LGBM_pred_train)*100,accuracy_score(Y_test,LGBM_pred_test)*100]]

Table_DF=pd.DataFrame(data_val,columns=["Model Name","Train Accuracy","Test Accuracy"],dtype=float)
print(Table_DF)

#Fitting model on entire data
Model_XGB.fit(X,Y)

#Exporting model to disk
pickle.dump(Model_XGB,open('Model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

