import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# column_names = ['ID', 'Diagnosis',
#                 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
#                 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
#                 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
#                 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
#                 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
#                 'fractal_dimension_worst']

# Load the data file (assuming you've downloaded it as 'wdbc.data')
# data_file = 'datasets\data.csv'
# print(os.path.isfile('data.csv'))
# print(os.listdir('.'))

# # Read the data into a pandas DataFrame
# df = pd.read_csv(data_file, header=None, names=column_names)
# output_csv = os.path.join('datasets', 'breast_cancer_data.csv')
# df.to_csv(output_csv, index=False)

# print(f"Data saved to {output_csv}")

# Load the data
data = pd.read_csv('datasets/hours_studied.csv')

# Split into independent and dependent variables
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Logistic regression object
class logistic_regression():
    def __init__(self, X_train, y_train, iterations=1000, learning_rate=0.015) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.X = X_train.to_numpy()
        self.y = y_train.to_numpy().reshape(-1, 1)
        self.rows = self.X.shape[0]
        self.col = self.X.shape[1]
        self.W = np.zeros((self.col, 1))
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, y_pred):
        return -(1 / self.rows) * np.sum(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))

    def train(self):
        costs = []
        dWs = []
        dbs = []
        y_preds = []
        iterations_list = []

        for i in range(self.iterations + 1):
            Z = self.X @ self.W + self.b
            y_pred = self.sigmoid(Z)

            cost = self.cost_function(y_pred)

            dW = (1 / self.rows) * (self.X.T @ (y_pred - self.y))
            db = (1 / self.rows) * np.sum(y_pred - self.y)
            
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db

            # Store cost, dW, db, and y_pred for visualization
            costs.append(cost)
            dWs.append(np.linalg.norm(dW))  # Use the norm of dW for simplicity
            dbs.append(db)
            y_preds.append(y_pred.mean())  # Store mean prediction
            iterations_list.append(i)

            if i % 100 == 0:
                print("Iteration", i, ": cost =", cost)

        # Plot results
        self.plot_results(costs, dWs, dbs, y_preds, iterations_list)

    def plot_results(self, costs, dWs, dbs, y_preds, iterations_list):
        # Plot cost over iterations
        plt.figure(figsize=(14, 10))

        # Subplot for cost
        plt.subplot(2, 2, 1)
        plt.plot(iterations_list, costs, label="Cost")
        plt.title("Cost Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)

        # Subplot for dW
        plt.subplot(2, 2, 2)
        plt.plot(iterations_list, dWs, label="dW")
        plt.title("dW (Gradient for Weights) Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Gradient Norm (dW)")
        plt.grid(True)

        # Subplot for db
        plt.subplot(2, 2, 3)
        plt.plot(iterations_list, dbs, label="db")
        plt.title("db (Gradient for Bias) Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Gradient (db)")
        plt.grid(True)

        # Subplot for sigmoid predictions (y_pred)
        plt.subplot(2, 2, 4)
        plt.plot(iterations_list, y_preds, label="Mean Sigmoid Output (y_pred)")
        plt.title("Sigmoid Output (Mean y_pred) Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Mean y_pred")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def test(self, X_test, y_test):
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy().reshape(-1, 1)

        Z_test = X_test @ self.W + self.b
        y_pred = self.sigmoid(Z_test)

        y_pred = y_pred > 0.5

        accuracy = np.sum(y_test == y_pred) / y_test.shape[0] * 100
        print("The accuracy of the model =", accuracy, "%")

# Initialize and train logistic regression model
log_regress = logistic_regression(X_train, y_train)
log_regress.train()

# Test the model on test data
log_regress.test(X_test, y_test)