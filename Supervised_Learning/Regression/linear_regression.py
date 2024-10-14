import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class linear_regression():
    def __init__(self, X_train, y_train, step_size = 0.05, iteration_count = 100000) -> None:
        self.step_size = step_size
        self.X_train = X_train
        self.y_train = y_train
        
        # best practice not to initialize weights to all zero at first
        self.weights = np.random.rand(X_train.shape[1])*0.01
        self.b0 = np.random.rand()*0.01
        self.iteration_count = iteration_count

    def mse(self, y_pred, y):
        return (1/y.shape[0]) * np.sum((y_pred - y)**2)

    def train(self):
        # y = b0 + x1*b1 + ... xn*bn
        # print(self.X_train.shape)
        # print(self.y_train.shape)
        # print(self.b0.shape)

        for i in range(self.iteration_count + 1):
            y_pred = self.X_train @ self.weights + self.b0
            # print(y_pred)
            
            self.b0 = self.b0 - self.step_size * (2/self.X_train.shape[0]) * np.sum(y_pred - self.y_train)
            self.weights = self.weights - self.step_size * (2/self.X_train.shape[0]) * (self.X_train.T @ (y_pred-self.y_train))

            if i % 1000 == 0:
                print("Iteration", i, ": cost =", self.mse(y_pred, self.y_train))

    def test(self, X_test, y_test):
        
        y_pred = X_test @ self.weights + self.b0

        mse = self.mse(y_pred,y_test)
        r2 = r2_score(y_test, y_pred)
        print("MSE =", mse)
        print("R2 =", r2)

if __name__ == "__main__":
    # This is a bunch object/data-type
    housing = fetch_california_housing()

    # print(housing.data)
    # print(housing.target)

    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['MedianHouseValue'] = housing.target

    print(data.head())

    # .values at the end converts from pandas to numpy array
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #split the data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=0)

    linregress = linear_regression(X_train,y_train)
    linregress.train()
    linregress.test(X_test,y_test)
