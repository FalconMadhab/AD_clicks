import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the advertising.csv file 

ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())


#Exploratory Data Analysis
#Histogram with Age
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
plt.show()  #Figure_1

#Jointplot showing Area Income versus Age.

sns.jointplot(x='Age',y='Area Income',data=ad_data)
plt.show()  #Figure_2

#Jointplot showing the kde distributions of Daily Time spent on site vs. Age

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');
plt.show()  #Figure_3
#Jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
plt.show()  #Figure_4

#Pairplot with the hue defined by the 'Clicked on Ad' column feature.

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
plt.show()  #Figure_5

#Logistic Regression
#Spliting the data into training set and testing set using train_test_split

from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Training and fitting the logistic regression model on the training set.
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Predictions and Evaluations
predictions = logmodel.predict(X_test)


#Classification report for the model.
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#Result of Classification Model
'''
precision    recall  f1-score   support

          0       0.87      0.96      0.91       162
          1       0.96      0.86      0.91       168

avg / total       0.91      0.91      0.91     

'''