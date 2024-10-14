import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class decision_tree_regressor():
    def __init__(self, min_samples_split=2, max_depth=3):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.complete_tree = None

    def fit(self, X, y):
        self.complete_tree = self.build_tree(X,y,depth=0)
        # print(self.complete_tree)
        return self.complete_tree

    def build_tree(self, X, y, depth):
        samples, features = X.shape

        if samples >= self.min_samples_split and depth < self.max_depth:
            best_feature, best_threshold, best_variance = self.best_split(X, y)
            # print(best_feature, best_threshold, best_variance)

            if best_variance > 0:
                left_index = X[:,best_feature] <= best_threshold
                right_index = X[:,best_feature] > best_threshold

                left_subtree = self.build_tree(X[left_index], y[left_index], depth+1)
                right_subtree = self.build_tree(X[right_index], y[right_index], depth+1)

                return {
                    'feature_index': best_feature,
                    'threshold': best_threshold,
                    'left': left_subtree,
                    'right': right_subtree
                }
            
        return np.mean(y)
    
    def best_split(self,X,y):
        best_feature = None
        best_threshold = None
        best_variance = float('-inf')

        samples, features = X.shape

        # Step 1 - we are working by column
        for index in range(features):
            feature_values = X[:, index]
            # print(feature_values)
            # this removes duplicate values from the array
            # possible_thresholds = np.unique(feature_values)
            sorted_values = np.sort(np.unique(feature_values))
            possible_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0  # Midpoints
        
            # Step 2 - we are working by row
            for threshold in possible_thresholds:
                # print(threshold)
                left_index = feature_values <= threshold
                right_index = feature_values > threshold
                # print(left_index)
                # print(right_index)

                left_threshold = y[left_index]
                right_threshold = y[right_index]
                
                if len(left_threshold) > 0 and len(right_threshold) > 0:
                    left_variance = np.var(left_threshold)
                    right_variance = np.var(right_threshold)

                    # Find variance reduction
                    variance_reduction = np.var(y) - (
                        (len(left_threshold)/len(y))*left_variance +
                        (len(right_threshold)/len(y))*right_variance
                    )

                    if variance_reduction > best_variance:
                        best_variance = variance_reduction
                        best_feature = index
                        best_threshold = threshold
        
        return best_feature, best_threshold, best_variance
    
    def predict(self, X):
        solution = np.array([self.predict_single(x, self.complete_tree) for x in X])
        return solution
    
    def predict_single(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature_index']] <= tree['threshold']:
                return self.predict_single(x, tree['left'])
            else:
                return self.predict_single(x, tree['right'])
        else:
            return tree


    def print_tree(self, tree, depth = 0):
        if isinstance(tree, dict):
            print(f"{'  '*depth}[X{tree['feature_index']} <= {tree['threshold']}]")
            print(f"{'  '*depth}--> Left:")
            self.print_tree(tree['left'],depth+1)
            print(f"{'  '*depth}--> Right:")
            self.print_tree(tree['right'],depth+1)
        else:
            print(f"{'  '*depth}Predicted Value = {tree}")

if __name__ == "__main__":

    # Built in scikit learn implemention
    data = pd.read_csv('datasets/decision_tree_dataset.csv')
    print(data.head())
    X = data.iloc[:,1:2].astype(int).values
    y = data.iloc[:,2].astype(int).values.reshape(-1,1)


    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X,y) 
    
    # My Code
    dec_regress = decision_tree_regressor(min_samples_split=2, max_depth=4)
    tree = dec_regress.fit(X,y)
    dec_regress.print_tree(tree)
    print(dec_regress.predict([[6500]]))

    X_grid = np.arange(min(X), max(X), 0.01)
    # converts to a column vector
    X_grid = X_grid.reshape((len(X_grid), 1)) 

    percent = 100 * (X.shape[0] - np.count_nonzero(dec_regress.predict(X) - regressor.predict(X))) / X.shape[0]

    plt.scatter(X,y, color='blue')
    plt.plot(X_grid, dec_regress.predict(X_grid), color='green', label='My Graph')
    plt.plot(X_grid, regressor.predict(X_grid), color='orange', label='Scitkit_Learn Graph')
    plt.title("Profit to Production Cost - Decision Tree Regression")
    plt.xlabel('Production Cost')
    plt.ylabel('Profit')
    plt.text(max(X)*.6, 0, f"Similiar = {percent: 0.2f}%", fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()