#This model is used to predit CO2EMISSION of the car based on other features like MAKE, MODEL, VEHICLE CLASS, CYLINDERS,
#CITY, HIGHWAY ETC

#Import required libraries
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

#Let us load the data
initial_df = pd.read_csv("FuelConsumption.csv")
#A check to see that data has been loaded properly
print (initial_df.head())

#Let us first see some details about the fields present
print (initial_df.describe())

#Let us do some visual analysis. Let us plot Cylinder vs Emission to see the relationship
plt.scatter(initial_df['CYLINDERS'],initial_df['CO2EMISSIONS'],color='Green')
plt.xlabel('Cylinders')
plt.ylabel('Emission')
#plt.show()
#We can clearly see the trend that CO2EMISSION increases with the number of CYLINDERS.

#Let us analyze CO2EMISSION a bit
print (initial_df['CO2EMISSIONS'].describe())

#Now let us try to implement linear regression to predict CO2EMISSIONS
print (initial_df.dtypes)

#We need to igonre Modelyear since all the data contains only 2014.
#We need to convert MAKE, MODEL, VEHICLECLASS, TRANSMISSION AND FUELTYPE to dummy variables
initial_df_with_dummmies = pd.get_dummies(initial_df)
print (initial_df_with_dummmies.dtypes)

#Now let us check if we have any missing data
print (initial_df_with_dummmies.isnull().sum().sort_values(ascending=False))
#Good that we don't have any missing data

#Now let us see the corelation between different variables
corrmat = initial_df_with_dummmies.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'CO2EMISSIONS')['CO2EMISSIONS'].index
cm = np.corrcoef(initial_df_with_dummmies[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#FUELCONSUMPTION_CITY/COMB AND HWY are highly correlated. We can keep only CITY to ensure model doesn't misbehave.
#Let us drop COMB and HWY columns from dataset
initial_df_with_dummmies = initial_df_with_dummmies.drop(initial_df_with_dummmies.columns[[0,4,5]],axis=1)

#Let us now have train/test split and build the model
#Let us shuffle the dataframe so that we don't end up getting unwanted patterns.
initial_df_with_dummmies = initial_df_with_dummmies.reindex(np.random.permutation(initial_df_with_dummmies.index))
#List of all the features
X = initial_df_with_dummmies.drop(initial_df_with_dummmies.columns[[7]],axis=1).values
#We have Survived column which is our label at 0th column.
Y = initial_df_with_dummmies['CO2EMISSIONS'].values

#Let us split the data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0)

#Let us print the shapes of X_Xrain and Y_Test to ensure we have sufficient data to train and test
print(X_Train.shape)
print(X_Test.shape)

#We can start fitting the data now
classifier = LinearRegression()
classifier.fit(X_Train, Y_Train)

#Now let us see the co-efficients of the model
print ("Co-efficients :",classifier.coef_)

#Now let us predict the training data
Y_Predicted = classifier.predict(X_Test)

#Let us see RMSE of the model
RMSE = sqrt(mean_squared_error(Y_Test, Y_Predicted))
print ("Root Mean Square Error = {}".format(RMSE))

#Now let us see Variance of the model
print('Variance score = {}'.format(classifier.score(X_Test, Y_Test)))