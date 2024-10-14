from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class l1_lasso_regression():
    def __init__(self, alpha=0.001, learning_rate=0.01, iterations=1000) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def cost(self, X, y):
        rows, cols = X.shape
        y_pred = X @ self.theta
        return (1 / (2*rows)) * np.sum((y-y_pred)**2) + self.alpha*np.sum(np.abs(self.theta))

    def fit(self, X, y):
        rows, cols = X.shape

        # initialize theta vectors with all zeros
        self.theta = np.zeros((cols,1))
        print(self.theta.shape)
        y_pred = X @ self.theta
        print(y_pred.shape)

        self.cost_history = []
        
        for i in range(self.iterations):
            y_pred = X @ self.theta
            gradient = -(1/rows) * (X.T @ (y-y_pred)) + self.alpha*np.sign(self.theta)

            self.theta = self.theta - self.learning_rate * gradient
            cost = self.cost(X,y)
            self.cost_history.append(cost)

    def predict(self,X):
        return X @ self.theta

if __name__ == '__main__':
    
    # Two different test case, set test_case to 0 or 1 to try either
    # 1 = built in Lasso function on housing dataset
    # 0 = "from scratch" Lasso function on simple polynomial function
    test_case = 1
    
    if(test_case):
        data = pd.read_csv('datasets/Melbourne_housing_FULL.csv')
        print(data.head())
        print(data.nunique())

        cols_to_use = [
            'Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
            'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price'
        ]

        data = data[cols_to_use]
        print(data.shape)

        # Shows me the rows that do not have values for each column
        print(data.isna().sum())

        cols_to_fill_zero = [
            'Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car'
        ]
        data[cols_to_fill_zero] = data[cols_to_fill_zero].fillna(0)

        # Landsize and BuildingArea still need to be dealt with
        data['Landsize'] = data['Landsize'].fillna(data.Landsize.mean())
        data['BuildingArea'] = data['BuildingArea'].fillna(data.BuildingArea.mean())
        print(data.isna().sum())

        # Simply drop the remaining NA values in each column
        data.dropna(inplace=True)
        print(data.isna().sum())

        # Text columns converted to values via OneHotEncoding, drop fist to prevent multicollinearity
        data = pd.get_dummies(data, drop_first=True)
        print(data.head())

        X = data.drop('Price', axis=1)
        y = data['Price']

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        # test on linear regression for comparison
        reg = LinearRegression().fit(X_train, y_train)
        print(reg.score(X_test, y_test))
        print(reg.score(X_train, y_train))

        lasso = Lasso(alpha=20, max_iter=500, tol=0.1)
        lasso.fit(X_train, y_train)
        print(lasso.score(X_test,y_test))
        print(lasso.score(X_train,y_train))  
    
    else:
    
        X = np.linspace(-1,1,200).reshape(-1,1)
        y = 1 - X +  2*X**3 + np.random.randn(200,1)*0.25
        plt.scatter(X,y,color='blue',label='Training Data',s=5)

        # Create polynomial features (degree 3 for cubic terms)
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)     

        # My Code
        lasso_reg = l1_lasso_regression()
        
        lasso_reg.fit(X_poly,y)
        y_pred = lasso_reg.predict(X_poly)

        plt.plot(X, y_pred, color='orange', label='L1 - Lasso', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title("Initial Graph")
        plt.legend()
        plt.grid(True)
        plt.show()


