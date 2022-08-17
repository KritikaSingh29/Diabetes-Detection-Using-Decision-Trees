#importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from tkinter import *
from sklearn import *
import joblib

#reading the csv file and giving each col heading:
col_names = ['Pregnant', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r"C:\Data\Amity\NTCC Sem 5\DiabetesDetectionUsingDecisionTrees\pima-indians-diabetes.csv",header = None, names = col_names)

#display top 5 rows of the dataset:
data.head()

#display last 5 rows of the dataset:
data.tail()

#find shape of our dataset:
print("Number of cols in dataset:", data.shape[0])
print("Number of rows in dataset:", data.shape[1])

#get info about our dataset like total rows, columns, datatypes of each col and memory requirement:
data.info()

#check null values :
data.isnull().sum()

#get overall stats of the dataset:
data.describe()
#creating a copy of the dataset to remove the zero val from the dataset:
data_copy = data.copy(deep = True)
data.columns
#replacing 0 values to np.nan
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.nan)
data_copy.isnull().sum()

#changing the np.nan values to mean of the col:
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())

feature_col = ['Pregnant', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#Store Feature Matrix in X and Response(Target) in Vector y
X = data[feature_col]
y = data.Outcome

#Splitting the datatset into the training set and the test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#Scikit-Learn Pipeline (Machine Learning Pipeline)

#1. using logistic regression
pipeline_lr = Pipeline([('scalar1', StandardScaler()), ('lr_classifier', LogisticRegression())])

#2. for kneighbors
pipeline_knn = Pipeline([('scalar2', StandardScaler()), ('knn_classifier', KNeighborsClassifier())])

#3. svc
pipeline_svc = Pipeline([('scalar3', StandardScaler()), ('svc_classifier', SVC())])

#4.decisionTress
pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier(criterion='entropy',max_depth = 3))])

#5.RandomForestClassifier
pipeline_rf = Pipeline([('rf_classifier', RandomForestClassifier())])

#6.GradientBoostingClassifier
pipeline_gbc = Pipeline([('gbc_classifier', GradientBoostingClassifier())])

pipelines = [pipeline_lr,
             pipeline_knn,
             pipeline_svc ,
             pipeline_dt,
             pipeline_rf,
             pipeline_gbc ]

#training our pipelines:
for pipe in pipelines:
    pipe.fit(X_train,y_train)

#see the accuracy of the models:
pipe_dict = {0:'LR',
             1:'KNN',
             2:'SVC',
             3:'DT',
             4:'RF',
             5:'GBC'}
for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)*100))

#using decison tree classifier:
X = data[feature_col]
y = data.Outcome
dt = DecisionTreeClassifier(max_depth = 3)
dt.fit(X,y)

#prediction on new data:
new_data = pd.DataFrame({
    'Pregnant':6,
    'Glucose':148.0,
    'BloodPressure':72.0,
    'SkinThickness':35.0,
    'Insulin':79.799479,
    'BMI':33.6,
    'DiabetesPedigreeFunction':0.627,
    'Age':50},index = [0])
p = dt.predict(new_data)
if p[0]==0:
    print('Non - Diabetic')
else:
    print('Diabetic')

#Saving model using joblib
joblib.dump(dt,'model_joblib_diabetes')
model = joblib.load('model_joblib_diabetes')

#GUI
def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())

    model = joblib.load('model_joblib_diabetes')
    result = model.predict([[p1,p2,p3,p4,p5,p6,p7,p8]])

    if result == 0:
        Label(master, text = "Non-Diabetic").grid(row  = 9,column =1)
    else:
        Label(master, text = "Diabetic").grid(row  = 9,column = 1)
master = Tk()
master.title("Diabetes Prediction Using Machine Learning")
label = Label(master, text = "Diabetes Prediction Using Machine Learning" , bg = "black", fg = "white").grid(row  = 0,columnspan = 2)
Label(master, text = "Pregnant").grid(row  = 1)
Label(master, text = "Glucose").grid(row  = 2)
Label(master, text = "Enter value of BloodPressure").grid(row  = 3)
Label(master, text = "Enter value of SkinThickness").grid(row  = 4)
Label(master, text = "Enter value of Insulin").grid(row  = 5)
Label(master, text = "Enter value of BMI").grid(row  = 6)
Label(master, text = "Enter value of DiabetesPedigreeFunction").grid(row  = 7)
Label(master, text = "Enter value of Age").grid(row  = 8)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.grid(row = 1, column = 1)
e2.grid(row = 2, column = 1)
e3.grid(row = 3, column = 1)
e4.grid(row = 4, column = 1)
e5.grid(row = 5, column = 1)
e6.grid(row = 6, column = 1)
e7.grid(row = 7, column = 1)
e8.grid(row = 8, column = 1)

b = Button (master, text = 'Predict', command = show_entry_fields).grid(row = 9, columnspan = 1)

master.mainloop()

