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
[exp 3 - Jupyter Notebook.pdf](https://github.com/user-attachments/files/19938760/exp.3.-.Jupyter.Notebook.pdf)

## X values:
![image](https://github.com/user-attachments/assets/bdec43e1-b96f-4482-b846-8486013f631a)



## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
