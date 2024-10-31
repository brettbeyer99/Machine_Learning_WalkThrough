from sklearn.datasets import make_regression, make_friedman1
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# goal is to minimize 1/2 ||w||^2 + C*summation(epsilon_i to epsilon_i*)
# subject to yi - (w*x +b) <= e + epsilon_i
#            (w*x +b) - yi <= e + epsilon_i*
#            epsilon_i*epsilon_i* >= 0

class SVR:
    def __init__(self, epsilon=.4, C=4.0, max_iterations=5000, learning_rate=0.01):
        self.epsilon = epsilon # this is the tube size with which predictions are not penalize
        self.C = C # Regularization parameter (controls weight of penalty)
        self.w = None # weight vector
        self.b = None # bias term
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def objective_function(self, pos_slack, neg_slack,):
        regularization = 0.5 * np.dot(self.w,self.w)
        penalty = self.C * np.sum(pos_slack + neg_slack)
        return regularization + penalty

    def fit(self, X, y):
        # initial parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0
        pos_slack = np.zeros(len(y))
        neg_slack = np.zeros(len(y))

        # Gradient Descent
        for i in range(self.max_iterations):
            gradient_w, gradient_b, gradient_positive_slack, gradient_negative_slack = self.gradient(pos_slack, neg_slack, X,y)

            # update values
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b
            pos_slack = np.maximum(0, pos_slack - self.learning_rate * gradient_positive_slack)
            neg_slack = np.maximum(0, neg_slack - self.learning_rate * gradient_negative_slack)

            if np.linalg.norm(gradient_w) < 1e-6 or abs(gradient_b) < 1e-6:
                break

    def gradient(self, pos_slack, neg_slack, X, y):
        gradient_w = np.zeros_like(self.w)
        gradient_b = 0
        gradient_positive_slack = np.ones(len(pos_slack)) * self.C
        gradient_negative_slack = np.ones(len(neg_slack)) * self.C

        for i in range(len(y)):
            error_val = y[i] - (np.dot(X[i], self.w) + self.b)
            if error_val > self.epsilon:
                gradient_b -= 1
                gradient_w -= X[i]
                gradient_positive_slack[i] = 1
            elif -error_val > self.epsilon:
                gradient_b += 1
                gradient_w += X[i]
                gradient_negative_slack[i] = 1

        return gradient_w, gradient_b, gradient_positive_slack, gradient_negative_slack

    def predict(self, X):
        return np.dot(X, self.w) + self.b


if __name__ == "__main__":
    
    # X,y = make_regression(n_samples=100, n_features=1, noise=25, random_state=10)
    np.random.seed(0)
    X = 2 * np.random.rand(100,1)
    y = 3 * X.squeeze() + np.random.randn(100)*0.5
    X= X.reshape(-1,1)

    sv_regress = SVR()
    sv_regress.fit(X,y)
    y_pred = sv_regress.predict(X)

    epsilon = sv_regress.epsilon
    y_upper = y_pred + epsilon
    y_lower = y_pred - epsilon

    plt.scatter(X, y, color="blue",label="Actual_Data")
    plt.plot(X, y_pred, color="red")
    plt.fill_between(X.squeeze(), y_upper, y_lower, color="orange",alpha=0.3)
    plt.show()

    # plt.scatter(X,y)
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.show()