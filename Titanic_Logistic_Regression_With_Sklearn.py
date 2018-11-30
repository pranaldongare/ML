import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

# Load training data set from CSV file
training_df_initial = pd.read_csv("train.csv")
training_df_initial = training_df_initial.drop(training_df_initial.columns[[0,3,8]],axis=1)
plt.rc("font",size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_style(style="white")
sns.set(style="whitegrid", color_codes=True)

#Check How many parameters have no values
print (training_df_initial.isnull().sum())

#Cabin has too many no values and won't be possible to extrapolate all of them
#hence we drop the Cabin column. Also, we don't think that column will add to the value
training_df_initial = training_df_initial.drop(['Cabin'],axis=1)

#Age has also large number of non values but we guess it will add to the regression and
#hence want to keep the column.
#To extrapolate the age, let us compare with Parch
age_to_parch_dataframe = training_df_initial[training_df_initial['Age'].isnull()]
print(age_to_parch_dataframe.shape)
sns.countplot(x='Parch',data=age_to_parch_dataframe,color="Blue")
plt.show()

#As above plot graph shows, majority of people don't have Parch associated.
#We will assume that these are middle aged person and will assign median value of the age data columns to them.

Age = training_df_initial['Age'].median()
for index, row in training_df_initial.iterrows():
    if np.isnan(row['Age']):
        training_df_initial.at[index,'Age'] = Age

print(training_df_initial.isnull().sum())
#With above print statement we validate that Age don't contain any non values now.
#Embarked has couple of non values, we can safely drop those rows as number is less
training_df_initial = training_df_initial.dropna()
#Print shape and null count of training_df_initial to ensure everything is right
print(training_df_initial.isnull().sum())
print(training_df_initial.shape)

#We need to create dummy variables for PClass, Sex, Embarked
training_df_initial_with_dummies = pd.get_dummies(data=training_df_initial,columns=['Pclass','Sex','Embarked'])
print(list(training_df_initial_with_dummies.columns))

#We will drop the Pclass, Sex and Embarked columns now
training_df_initial_with_dummies = training_df_initial_with_dummies.drop(training_df_initial_with_dummies.columns[[1,2,7]],axis=1)
print(list(training_df_initial_with_dummies.columns))

#Now we are all set to define our features and labels, but before that let's find corelation between variables
sns.heatmap(training_df_initial_with_dummies.corr())
plt.show()

#I think we can use all the variables with current co-relation matrix. Let us proceed.
#Let us shuffle the dataframe so that we don't end up getting unwanted patterns.
training_df_initial_with_dummies = training_df_initial_with_dummies.reindex(np.random.permutation(training_df_initial_with_dummies.index))
#List of all the features
X = training_df_initial_with_dummies.iloc[:,1:]
#We have Survived column which is our label at 0th column.
Y = training_df_initial_with_dummies.iloc[:,0]

#Split the data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0)
#Let us print the shapes of X_Xrain and Y_Test to ensure we have sufficient data to train and test
print(X_Train.shape)
print(Y_Test.shape)

#We can start fitting the data now
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train, Y_Train)

#Let us predict and print the confusion matrix to see how good our model is
Y_Pred = classifier.predict(X_Test)

#Confusion matrix can be print using Sklearn functionality
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_Test, Y_Pred)
print(confusion_matrix)

#Let us see the Accuracy of our model
Accuracy = classifier.score(X_Test,Y_Test)
print('Accuracy {:.2f}'.format(Accuracy))

#Accuracy is 81% which is good enough on limited data set we have.

#Let us print Classification Report to get more insights into the model
from sklearn.metrics import classification_report
print(classification_report(Y_Test,Y_Pred))
