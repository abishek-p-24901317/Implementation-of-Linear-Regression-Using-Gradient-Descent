# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import all the required packages.

2. Display the output values using graphical representation tools as scatter plot and graph.

3. Predict the values using predict() function.

4. Display the predicted values and end the program.

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
![Screenshot 2025-03-10 105815](https://github.com/user-attachments/assets/5fcb5f42-05d0-40d2-9865-b9f238fe8220)
![Screenshot 2025-03-10 105959](https://github.com/user-attachments/assets/17ab154a-54ea-4f14-ba87-4a85b2260e72)
![Screenshot 2025-03-10 110033](https://github.com/user-attachments/assets/49b11261-b50d-41d2-b5c0-3c4a4c061ba5)
![Screenshot 2025-03-10 110054](https://github.com/user-attachments/assets/640693a4-2f4f-4472-8865-74b50d006418)
![Screenshot 2025-03-10 110112](https://github.com/user-attachments/assets/4b20864a-7eb9-4e6b-8ab7-ad632371c869)
![Screenshot 2025-03-10 110132](https://github.com/user-attachments/assets/7c08bf08-3f84-49d1-889e-9584f2e04545)
![Screenshot 2025-03-10 110153](https://github.com/user-attachments/assets/155175a6-10b8-4737-80f0-81d24da6f694)
![Screenshot 2025-03-10 110216](https://github.com/user-attachments/assets/73df3933-5b54-4c88-aa14-5e8085720fc5)
![Screenshot 2025-03-10 110240](https://github.com/user-attachments/assets/3d57a195-a4ef-43ef-850b-a54316a0327e)
![Screenshot 2025-03-10 110256](https://github.com/user-attachments/assets/1170a9a2-f2ef-4042-8d01-6f7d5be1142c)
![Screenshot 2025-03-10 110314](https://github.com/user-attachments/assets/d140cd1b-afce-4d7f-90c1-a77da03f8ee3)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
