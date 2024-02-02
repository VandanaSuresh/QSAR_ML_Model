#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import sklearn
import sys
import xgboost as xg 
from scipy.io import arff
from sklearn.svm import SVR
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from numpy import sqrt
import sklearn.metrics as metrics
import scipy.stats
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

  ##load input file

df = pd.read_csv('/home2/QSAR_check/weka_processing/final_features/weka_out_padel_filtered_mmff94_SW1417.csv')
    #thisFilter = df.filter("''")
    #df = df.drop(thisFilter, axis=1)
x= df.drop('IC50', axis=1)
y= df.IC50

 ##train-test splitting (80-20)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
x_train.shape,y_train.shape,y_test.shape,x_test.shape
x_test.replace([np.inf, -np.inf], 0, inplace=True)
x_train.replace([np.inf, -np.inf], 0, inplace=True)

#####################################
######## Regression Models ######### 
####################################

#cross-validation
cv= KFold(n_splits=10, random_state=1,shuffle=True)


# 1.random forest regression model #
rf = RandomForestRegressor(n_estimators=320,random_state=70).fit(x_train,y_train)
rf_pred_train = cross_val_predict(rf,x_train,y_train,cv=cv, n_jobs=-1)
rf_pred_test = rf.predict(x_test)

# 2. linear regression model #

lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
lr_pred_train = cross_val_predict(lr,x_train,y_train,cv=cv, n_jobs=-1)
lr_pred_test = lr.predict(x_test)

# 3.multiple linear regression model #
mlr = LinearRegression()
mlr.fit(x_train,y_train)
mlr_pred_train = cross_val_predict(mlr,x_train,y_train,cv=cv, n_jobs=-1)
mlr_pred_test = mlr.predict(x_test)

# 4.Polynominal Regression model #
poly_reg = PolynomialFeatures(degree = 1)
preg = LinearRegression()
pftr = poly_reg.fit_transform(x_train)
pfts = poly_reg.fit_transform(x_test)
preg.fit(pftr,y_train)
pr_pred_train = cross_val_predict(preg,x_train,y_train,cv=cv, n_jobs=-1)
pr_pred_test = preg.predict(pfts)

# 5.XGBoost Regression model #
xgb_r = xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, random_state=5)
xgb_r.fit(x_train, y_train)
xg_pred_train = cross_val_predict(xgb_r,x_train,y_train,cv=cv, n_jobs=-1)
xg_pred_test = xgb_r.predict(x_test)

# 6. Lasso Regression #
lasso = linear_model.Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
ls_pred_train = cross_val_predict(lasso,x_train,y_train,cv=cv, n_jobs=-1)
ls_pred_test = lasso.predict(x_test)

# 7. Gradient Boosting Regressor #
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)
gbr_params = {'n_estimators': 300,'max_depth': 5,'min_samples_split': 5,'learning_rate': 0.01,'loss': 'ls'}
gbr = GradientBoostingRegressor(**gbr_params)
gbr.fit(x_train_std, y_train)
gbr_pred_train = cross_val_predict(gbr,x_train_std,y_train,cv=cv, n_jobs=-1)
gbr_pred_test = gbr.predict(x_test_std)

# 8. Support Vector Regression #
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)
regr = SVR(kernel= 'linear',degree=0)
regr.fit(x_train, y_train)
svr_pred_train = cross_val_predict(regr,x_train_std,y_train,cv=cv, n_jobs=-1)
svr_pred_test = regr.predict(x_test_std)

# 9. Multilayer Perceptron
mlp = MLPRegressor(hidden_layer_sizes=(360,250,290),max_iter =280,activation='relu',solver='adam',random_state=2)
mlp.fit(x_train, y_train)
mlp_pred_train = cross_val_predict(mlp,x_train,y_train,cv=cv, n_jobs=-1)
mlp_pred_test = mlp.predict(x_test)

###training result####

#random forest#
corr_rf_tr, _ = pearsonr(y_train, rf_pred_train)
r2_rf_tr= metrics.r2_score(y_train, rf_pred_train)
mse_rf_tr = mean_squared_error(y_train, rf_pred_train)
rmse_rf_tr= mse_rf_tr**.5

#linear regression#
corr_lr_tr, _ = pearsonr(y_train, lr_pred_train)
r2_lr_tr = metrics.r2_score(y_train, lr_pred_train)
mse_lr_tr = mean_squared_error(y_train, lr_pred_train)
rmse_lr_tr = mse_lr_tr**.5

#MultiLinearRegression#
corr_mlr_tr, _ = pearsonr(y_train, mlr_pred_train)
r2_mlr_tr = metrics.r2_score(y_train, mlr_pred_train)
mse_mlr_tr = mean_squared_error(y_train, mlr_pred_train)
rmse_mlr_tr = mse_mlr_tr**.5

#PolynominalRegression#
corr_pr_tr, _ = pearsonr(y_train, pr_pred_train)
r2_pr_tr= metrics.r2_score(y_train, pr_pred_train)
mse_pr_tr = mean_squared_error(y_train, pr_pred_train)
rmse_pr_tr= mse_pr_tr**.5

#XGBoost#
corr_xg_tr, _ = pearsonr(y_train, xg_pred_train)
r2_xg_tr= metrics.r2_score(y_train, xg_pred_train)
mse_xg_tr = mean_squared_error(y_train, xg_pred_train)
rmse_xg_tr= mse_xg_tr**.5

#LassoRegression#
corr_ls_tr, _ = pearsonr(y_train, ls_pred_train)
r2_ls_tr= metrics.r2_score(y_train, ls_pred_train)
mse_ls_tr = mean_squared_error(y_train, ls_pred_train)
rmse_ls_tr= mse_ls_tr**.5

#GradientBoostingRegression#
corr_gbr_tr, _ = pearsonr(y_train, gbr_pred_train)
r2_gbr_tr= metrics.r2_score(y_train, gbr_pred_train)
mse_gbr_tr = mean_squared_error(y_train, gbr_pred_train)
rmse_gbr_tr= mse_gbr_tr**.5

#SupportVectorRegression#
corr_svr_tr, _ = pearsonr(y_train, svr_pred_train)
r2_svr_tr= metrics.r2_score(y_train, svr_pred_train)
mse_svr_tr = mean_squared_error(y_train, svr_pred_train)
rmse_svr_tr= mse_svr_tr**.5

#MultilayerPerceptron
corr_mlp_tr, _ = pearsonr(y_train, mlp_pred_train)
r2_mlp_tr= metrics.r2_score(y_train, mlp_pred_train)
mse_mlp_tr = mean_squared_error(y_train, mlp_pred_train)
rmse_mlp_tr= mse_rf_tr**.5

###testing results###
#rf#
corr_rf_ts,_ = pearsonr(y_test, rf_pred_test)
r2_rf_ts = metrics.r2_score(y_test, rf_pred_test)
mse_rf_ts = mean_squared_error(y_test, rf_pred_test)
rmse_rf_ts = mse_rf_ts**.5

#LR#
corr_lr_ts, _ = pearsonr(y_test, lr_pred_test)
r2_lr_ts = metrics.r2_score(y_test, lr_pred_test)
mse_lr_ts = mean_squared_error(y_test, lr_pred_test)
rmse_lr_ts = mse_lr_ts**.5

#MLR#
corr_mlr_ts, _ = pearsonr(y_test, mlr_pred_test)
r2_mlr_ts = metrics.r2_score(y_test, mlr_pred_test)
mse_mlr_ts = mean_squared_error(y_test, mlr_pred_test)
rmse_mlr_ts = mse_mlr_ts**.5

#PR
corr_pr_ts,_ = pearsonr(y_test, pr_pred_test)
r2_pr_ts = metrics.r2_score(y_test, pr_pred_test)
mse_pr_ts = mean_squared_error(y_test, pr_pred_test)
rmse_pr_ts = mse_pr_ts**.5

#XGB
corr_xg_ts,_ = pearsonr(y_test, xg_pred_test)
r2_xg_ts = metrics.r2_score(y_test, xg_pred_test)
mse_xg_ts = mean_squared_error(y_test, xg_pred_test)
rmse_xg_ts = mse_xg_ts**.5

#LASSO#
corr_ls_ts,_ = pearsonr(y_test, ls_pred_test)
r2_ls_ts = metrics.r2_score(y_test, ls_pred_test)
mse_ls_ts = mean_squared_error(y_test, ls_pred_test)
rmse_ls_ts = mse_ls_ts**.5

#GBR#
corr_gbr_ts,_ = pearsonr(y_test, gbr_pred_test)
r2_gbr_ts = metrics.r2_score(y_test, gbr_pred_test)
mse_gbr_ts = mean_squared_error(y_test, gbr_pred_test)
rmse_gbr_ts = mse_gbr_ts**.5

#SVR
corr_svr_ts,_ = pearsonr(y_test, svr_pred_test)
r2_svr_ts = metrics.r2_score(y_test, svr_pred_test)
mse_svr_ts = mean_squared_error(y_test, svr_pred_test)
rmse_svr_ts = mse_svr_ts**.5

#MLP
corr_mlp_ts,_ = pearsonr(y_test, mlp_pred_test)
r2_mlp_ts = metrics.r2_score(y_test, mlp_pred_test)
mse_mlp_ts = mean_squared_error(y_test, mlp_pred_test)
rmse_mlp_ts = mse_mlp_ts**.5

##########saving training and testing output in model_output.txt file########
file_path = 'model_output.txt'
sys.stdout = open(file_path, "w")

print(' ')
print(' ')
print('TRAINING RESULTS :')
print(' ')
print("Models\t\tR\t\tR^2\t\tMAE\t\tRMSE")
print("RFR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_rf_tr,r2_rf_tr,mse_rf_tr,rmse_rf_tr))
print("LR \t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_lr_tr,r2_lr_tr,mse_lr_tr,rmse_lr_tr))
print("MLR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_mlr_tr,r2_mlr_tr,mse_mlr_tr,rmse_mlr_tr))
print("PR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_pr_tr,r2_pr_tr,mse_pr_tr,rmse_pr_tr))
print("XGB\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_xg_tr,r2_xg_tr,mse_xg_tr,rmse_xg_tr))
print("LASSO.R\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_ls_tr,r2_ls_tr,mse_ls_tr,rmse_ls_tr))
print("GBR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_gbr_tr,r2_gbr_tr,mse_gbr_tr,rmse_gbr_tr))
print("SVR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_svr_tr,r2_svr_tr,mse_svr_tr,rmse_svr_tr))
print("MLP\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_mlp_tr,r2_mlp_tr,mse_mlp_tr,rmse_mlp_tr))

print(' ')
print(' ')
print('TESTING RESULTS :')
print(' ')
print("Models\t\tR\t\tR^2\t\tMAE\t\tRMSE")
print("RFR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_rf_ts,r2_rf_ts,mse_rf_ts,rmse_rf_ts))
print("LR \t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_lr_ts,r2_lr_ts,mse_lr_ts,rmse_lr_ts))
print("MLR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_mlr_ts,r2_mlr_ts,mse_mlr_ts,rmse_mlr_ts))
print("PR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_pr_ts,r2_pr_ts,mse_pr_ts,rmse_pr_ts))
print("XGB\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_xg_ts,r2_xg_ts,mse_xg_ts,rmse_xg_ts))
print("LASSO.R\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_ls_ts,r2_ls_ts,mse_ls_ts,rmse_ls_ts))
print("GBR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_gbr_ts,r2_gbr_ts,mse_gbr_ts,rmse_gbr_ts))
print("SVR\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_svr_ts,r2_svr_ts,mse_svr_ts,rmse_svr_ts))
print("MLP\t\t{0:.3f}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}".format(corr_mlp_ts,r2_mlp_ts,mse_mlp_ts,rmse_mlp_ts))


######################################################################
############### CREDITS TO KAVITA KUNDAL #############################
######################################################################


