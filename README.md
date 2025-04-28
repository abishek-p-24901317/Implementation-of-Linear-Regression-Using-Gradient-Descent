# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import necessary packages: numpy, pandas, and StandardScaler from sklearn.preprocessing.

2.Define linear_regression() to perform gradient descent and learn model parameters.

3.Load the dataset, extract features and target, convert data types, and standardize them.

4.Train the model using linear_regression() and obtain optimized parameters.

5.Standardize new input data, predict the output, inverse transform the result, and display the predicted value.

6.End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Abishek P
RegisterNumber: 212224240002
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    ##coefficient of b
    x=np.c_[np.ones(len(x1)),x1]
    ##initialize theta with zero
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    ##perform gradient decent
    for _ in range(num_iters):
        ##calculate predictions
        predictions=(x).dot(theta).reshape(-1,1)
        ##calculate errors
        errors=(predictions - y).reshape(-1,1)
        ##update theta using gradient descent
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
##assume the last column as your target varible y
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
##learn model parameters
theta=linear_regression(x1_scaled,y1_scaled)
##predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
## DATA HEAD:
![Screenshot 2025-04-28 160846](https://github.com/user-attachments/assets/f6a6d25e-46f3-4d7a-b1f8-c2bc7c85b2c3)
## X values:
![Screenshot 2025-04-28 213638](https://github.com/user-attachments/assets/2c071c13-1fae-4bc7-89f1-d679b348bef7)
## Y values:
![Screenshot 2025-04-28 213714](https://github.com/user-attachments/assets/8cd4160f-42a7-4f17-9ce4-6ed429327cdb)
## X1_scaled:
![Screenshot 2025-04-28 213853](https://github.com/user-attachments/assets/fd782239-568e-4ee7-afd9-4d2f1c04b735)
## Y1_scaled:
![Screenshot 2025-04-28 213928](https://github.com/user-attachments/assets/1d91c223-328d-4f74-b662-ed0929e9f735)
## Predicted Value:
![Screenshot 2025-04-28 214016](https://github.com/user-attachments/assets/2432cf68-f03c-401c-8353-5d42ae62aede)
## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
