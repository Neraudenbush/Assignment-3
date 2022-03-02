# Assignment-3
```Python
#Import packages and define kernals
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae


def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)


def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  ```
  ```Python
  #Define lowess regression function
def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
    
    def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
  #Extreme Gradient Boosting
   model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    ```
    ```Python
    #Test it on the Cars data
cars = pd.read_csv('cars.csv')

X = cars[['ENG','CYL','WGT']].values
y = cars['MPG'].values

kf = KFold(n_splits=10,shuffle=True,random_state=410)
scale = StandardScaler()
```
```Python
mse_lwr = []
mse_blwr = []
mse_xgb = []
mae_lwr = []
mae_blwr = []
mae_xgb = []

# this is the Cross-Validation Loop for the cars data
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  mse_xgb.append(mse(ytest,yhat_xgb))
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mae_xgb.append(mae(ytest,yhat_xgb))
  mae_lwr.append(mae(ytest,yhat_lwr))
  mae_blwr.append(mae(ytest,yhat_blwr))
print('The Cross-validated Mean Squared Error for LWR on cars is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Absolute Error for LWR on cars is : '+str(np.mean(mae_lwr)))
print()
print('The Cross-validated Mean Squared Error for BLWR on cars is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Absolute Error for BLWR on cars is : '+str(np.mean(mae_blwr)))
print()
print('The Cross-validated Mean Squared Error for XGB on cars is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Absolute Error for XGB on cars is : '+str(np.mean(mae_xgb)))
```
