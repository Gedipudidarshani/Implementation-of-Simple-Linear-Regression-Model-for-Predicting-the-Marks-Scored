# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gedipudi Darshani
RegisterNumber:212223230062
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
```
dataset.info()
```
```
x=dataset.iloc[:,:-1].values
print(x)
y =dataset.iloc[:,-1].values
print(y)
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
x_train.shape
x_test.shape
```
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```
```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/40aa0093-96a9-4194-a80f-875b7cd02238)
![image](https://github.com/user-attachments/assets/473c6db2-9678-4791-999d-fede26d7aae0)
![image](https://github.com/user-attachments/assets/3358e667-a1a5-4a04-b7db-8d6be3080ef6)
![image](https://github.com/user-attachments/assets/e3d7fda9-be2c-4ac5-948f-424356de6cad)
![image](https://github.com/user-attachments/assets/981b2e22-e7aa-4e18-9580-c1e6f2297fef)
![image](https://github.com/user-attachments/assets/9e642509-0f0e-4267-b093-f44d71fd9968)
![image](https://github.com/user-attachments/assets/d8ce2355-db06-4014-8855-a5035d619c57)
![image](https://github.com/user-attachments/assets/39cac148-6547-49d5-8614-7f6b018cd355)
![image](https://github.com/user-attachments/assets/220916aa-696d-49ee-bbcb-bf2a1f5cf6b2)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
