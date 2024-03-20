# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### Step-1 
Import the required packages and print the present data.
### Step-2
Print the placement data and salary data.
### Step-3
Find the null and duplicate values.
### Step-4
Using logistic regression find the predicted values of accuracy , confusion matrices.
### Step-5
Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANTHOSH.T
RegisterNumber:212223220100  
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0 )

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
classification_report1 =classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
The Logistic Regression Model to Predict the Placement Status of Student:
## Placement Data:
![Screenshot 2024-03-20 113321](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/92d9f627-c903-4329-a6fe-c3fd4653fede)
## Y_prediction array:
![Screenshot 2024-03-20 113353](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/674b9a2e-b2c9-44b2-b970-0ebddc1f81fc)
## Accuracy value:
![Screenshot 2024-03-20 113406](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/56f17194-7cca-46b2-ab56-0323251b1d5a)
## Confusion array:
![Screenshot 2024-03-20 113414](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/f80d7eab-00e4-45af-8ccf-678454f198d0)

## Classification Report:
![Screenshot 2024-03-20 113449](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/616d9723-e7be-411b-91cf-488b414f0a88)
## Prediction of LR:
![Screenshot 2024-03-20 113520](https://github.com/SanthoshThiru/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148958618/6cf6849b-0d45-4d56-b589-1ec3842c4189)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
