# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import (
    tree,
    model_selection,
    metrics,
    preprocessing,

)
#ignore warnings
import warnings
warnings.filterwarnings('ignore')



# Read and Visualize data
df = pd.read_csv('heart.csv')
df.head()
df.select_dtypes(include=['object']).columns

# Checking for NULLs in the data
df.isnull().sum()

# Label encoding for classied column
labels=["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
label_encoder = preprocessing.LabelEncoder()
for label in labels: 
    df[label]= label_encoder.fit_transform(df[label])
    
# Spliting dataset for test and train purpose 75% for training and 25% for testing

y = df[['HeartDisease']]
del df['HeartDisease']
x = df
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

### Decision Tree ####.............................................................................

# Declaring/ Defining Decision Tree Classifier
tModel = tree.DecisionTreeClassifier(max_depth=4,max_leaf_nodes=14)

# fit the model using train dataset
tModel.fit(x_train,y_train)

# Plot figure for trr model
plt.figure(figsize=(20, 20), dpi=450)
tree.plot_tree(tModel, filled=True)
plt.savefig('tree.png')

# Print the accuracy using / testing by test dataset
print("\nDecision Tree Model Accuracy is {} %".format(tModel.score(x_test,y_test)*100))
print("--------------------------------------------------\n")

# checking the score a
pred = tModel.predict(x_test)
print("Classification_Report(DT): \n", metrics.classification_report(pred,y_test))

# make another confusion matrix with sns.heatmap
cm = confusion_matrix(y_test, pred)
fig , ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm, annot=True, ax=ax , fmt='d')
ax.set_xlabel("predicted" , fontsize=20)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticklabels(['0', '1'], fontsize = 10)
ax.xaxis.tick_top()
plt.title("DT Confusion matrix", fontsize=17, color='red')

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1'], fontsize = 10)
plt.show()

print("Confusion Matrix (DT): \n", cm)

## Logistic Regression ...........................................................................

# Scalling the data using standard scaler
scaler=StandardScaler() 
scaler.fit(x_train) 
x_train_scaler=scaler.transform(x_train) 
scaler=StandardScaler()
scaler.fit(x_train)
x_test_scaler=scaler.transform(x_test)

# Declaring model for logistic regression
logisticModel = LogisticRegression()

# Fit the model using train dataset
logisticModel.fit(x_train_scaler,y_train.values.ravel())

# #Plotting coefficient of Logistic regression 
coefficient = logisticModel.coef_
plt.bar([col for col in x_train.columns], coefficient [0])
plt.title("Coefficients")

plt.xticks(
    rotation=40,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
plt.show()

# Printing accuracy using test dataset
print("\n\nLogistic Regression Model Accuracy is {}%". format (logisticModel.score(x_test_scaler,y_test)*100))

print("--------------------------------------------------------\n")

# checking the score a
pred2 = logisticModel.predict(x_test_scaler)
print("Classification_Report(LR): \n", metrics.classification_report(pred2,y_test))

# make another confusion matrix with sns.heatmap
cm2 = confusion_matrix(y_test, pred2)
fig , ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm2, annot=True, ax=ax , fmt='d')
ax.set_xlabel("predicted" , fontsize=20)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticklabels(['0', '1'], fontsize = 10)
ax.xaxis.tick_top()
plt.title("LR Confusion matrix", fontsize=17, color='red')

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1'], fontsize = 10)
plt.show()

print("Confusion Matrix (LR): \n", cm2)

## Naive Bayes Classifier ...........................................................................

# Defining the Gaussian Naive Bayes Model
gnb = GaussianNB()

# Fit the model using train dataset
gnb.fit(x_train,y_train.values.ravel())

# printing accuracy using test dataset
print("\n\nNaive Bayes Model Accuracy is {} %".format(gnb.score(x_test,y_test.values.ravel())*100))
print("--------------------------------------------------------")
pred3 = gnb.predict(x_test)
cm3 = confusion_matrix(y_test, pred3)
print("Confusion Matrix (GNB): \n", cm3)

### compare all classification models................................................................

models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

# Decision Trees
models['Decision Trees'] = DecisionTreeClassifier(criterion='gini', max_depth=4)

# Naive Bayes
models['Naive Bayes'] = GaussianNB()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Accuracy, Precision, Recall, F1= {}, {}, {}, {}

for key in models.keys():
    
    # Fit the classifier model
    models[key].fit(x_train, y_train)
    
    # Prediction 
    predictions = models[key].predict(x_test)
    
    # Calculate Accuracy, Precision and Recall Metrics
    Accuracy[key] = accuracy_score(predictions, y_test)
    Precision[key] = precision_score(predictions, y_test)
    Recall[key] = recall_score(predictions, y_test)
    F1[key] = f1_score(predictions, y_test)
    
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
df_model['Accuracy'] = Accuracy.values()
df_model['Precision'] = Precision.values()
df_model['Recall'] = Recall.values()
df_model['F1 Score'] = F1.values()

df_model.sort_values(by='Accuracy',ascending=False).style.background_gradient('twilight')

# ............................ END ......................................................................................
