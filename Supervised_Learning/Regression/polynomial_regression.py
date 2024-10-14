import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

class polynomial_regression():
    def __init__(self,degree=5) -> None:
        self.degree = degree
        self.design_matrix = None
        self.beta = None

    def fit(self, X_train, y_train):
        self.design_matrix = np.ones((X_train.shape[0], self.degree + 1))
        print(self.design_matrix)

        for i in range(1,self.degree+1):
            print((X_train ** i).flatten())
            self.design_matrix[:,i] = (X_train ** i).flatten()
            print(self.design_matrix)

        self.beta = np.linalg.inv((self.design_matrix.T @ self.design_matrix)) @ self.design_matrix.T @ y_train
        print(self.beta)

    def show(self, X, y):
        y_sol = self.design_matrix @ self.beta
        plt.scatter(X,y,color='blue',label='Training Data')
        plt.plot(X,y_sol,color='red',label=f'Polynomial Degree {self.degree} fit')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title("Polynomial Regression")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # First test set
    # data = pd.read_csv("datasets/polynomial_dataset.csv")
    # X = data.iloc[:,:-1].values
    # y = data.iloc[:,-1].values.reshape(-1,1)

    # Second test set
    X = np.linspace(-1,1,200).reshape(-1,1)
    y = 1 - X +  2*X**3 + np.random.randn(200,1)*0.25
    
    plt.scatter(X,y,color='blue',label='Training Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title("Initial Graph")
    plt.legend()
    plt.grid(True)
    plt.show()

    degree = 5

    poly_regress = polynomial_regression(degree)
    poly_regress.fit(X,y)
    poly_regress.show(X,y)