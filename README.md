# Prediction Using Supervised model (linear regression)

In this regression Project we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression Project as it involves just two variables.


## Step 1: Importing the required libraries

```python
#Importing the required libraries for the gives task
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
}
```
## Step 2: Importing the data
```python
#importing the data required for the task 
url = "http://bit.ly/w-data"
task_data=pd.read_csv(url)
task_data.head(10)
}
```
## Step 3: Visualization of the data
```python 
#visualization of the dataset
task_data.plot(kind="scatter", x="Hours",y="Scores") }
![download](https://user-images.githubusercontent.com/47120501/172880899-f573238a-2f1d-41d7-8983-1b0cfddda0b2.png)

Observation: Here we can see that the data already shows a linear behaviour or the scores and no. of hours studied are proportionate to each other.
```
## We use pandas dataframe for the iloc funtion which helps in slicing the data
```jupternotebook
X = task_data.iloc[:, :-1].values 
Y = task_data.iloc[:, 1].values

Here we used iloc function to slice the data so that we can have inputs and outputs, In order to split the data into the train and test datsets.
.values will help convert the data to object
```
## Step 4:Train the Model
```python# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
```
## Step 5:Making Predictions.
```jupternotebook
Y_pred = regressor.predict(X_test) # Predicting the scores
print(Y_pred)
[16.88414476 33.73226078 75.357018   26.79480124 60.49103328]
print(Y_test)
[20 27 69 30 62]
pd.DataFrame({"Actual" :Y_test,"predicted" :Y_pred})
```

## Step 6:Task question-What will be the predicted score if the student studies for 9.5 hours/day

```jupternotebook
hours_studied= 9.25
hours_studied=np.array(hours_studied)
hours_studied=hours_studied.reshape(-1,1)
own_pred = regressor.predict(hours_studied)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
```


## Step 7:Evaluating the model.
```jupternotebook
#using RMSE(root mean square error)
from sklearn.metrics import mean_squared_error
import math
actual = Y_test
predicted = Y_pred
MSE = mean_squared_error(actual,predicted)
rmse = math.sqrt(MSE)
print(rmse)
```
```jupternotebook
#using MSE(mean squared error)
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred))
