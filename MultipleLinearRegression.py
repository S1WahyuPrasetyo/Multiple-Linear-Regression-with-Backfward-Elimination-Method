# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, -1].values
y = dataset.iloc[:, -1].values

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # [0] merupakan indek kolom ke-0
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

#Splitting Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#X_train = X_train.reshape(40,1)
#X_test = X_test.reshape(10,1)
#y_train = y_train.reshape(40,1)
#y_test = y_test.reshape(10,1)

#Training the MOdel
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Predicx the result
y_pred = model.predict(X_test)
accuracy = model.score(X_train, y_train)
print(accuracy)

# Predicting the Test set results
y_pred = model.predict(X_test)
np.set_printoptions(precision=2) #Mengindari Jebakan variable dummmy dg mengubah VD menjadi 2 kolom
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



### Building the Optimal Model Using Backward Elimination
import statsmodels.api as sm
X= np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1) 
X_opt= X[:, [0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Remove 5
X_opt= X[:, [0,1,2,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Remove 2
X_opt= X[:, [0,1,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Remove 4
X_opt= X[:, [0,1,3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Get Result : R&D Spend and Marketing spend is the indepentdent variable highest  impact on profit of these startups