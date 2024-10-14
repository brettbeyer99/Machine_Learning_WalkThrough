# Decision trees are highly sensative to training data
# A random forest tree is much less sensative...

# Bootstrapping = random sampling with replacement
# Aggregation = looking at the results of multiple models and creating a solution
# Baggie = Bootstrapping + Aggregation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from decision_tree_regression import decision_tree_regressor

class random_forest_regessor():
    def __init__(self, min_samples_split = 2, max_depth = 3, num_trees = 5, feature_split_type = "sqrt"):
        self.num_trees = num_trees
        self.feature_split_type = feature_split_type
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.forest = []

    def fit(self, X, y):
        num_features = X.shape[1]
        self.num_features_to_split = self.determine_feature_split(num_features)

        for _ in range(self.num_trees):
            tree = decision_tree_regressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            new_X, new_y = self.bootstrap(X,y)
            tree.fit(new_X,new_y)
            self.forest.append(tree)


    def determine_feature_split(self, num_features):
        if self.feature_split_type == "all":
            return num_features
        elif self.feature_split_type == "sqrt":
            return int(np.sqrt(num_features) )
        elif self.feature_split_type == "log2":
            return int(np.log2(num_features))
        elif isinstance(self.feature_split_type, int):
            if self.feature_split_type <= num_features:
                return self.feature_split_type
            else:
                raise ValueError(f"Feature split type = {self.feature_split_type}, exceeds the number of features. Must choose a smaller value.")
        else:
            raise ValueError(f"{self.feature_split_type} is not an acceptable values. Must choose all, sqrt, log2, or an integer value.")
    
    def bootstrap(self, X, y):
        idxs_rows = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        idxs_features = np.random.choice(X.shape[1], size=self.num_features_to_split, replace=True)
        return X[idxs_rows][:, idxs_features], y[idxs_rows]
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.forest])
        return np.mean(tree_predictions, axis=0)

if __name__ == "__main__":

    # Built in scikit learn implemention
    data = pd.read_csv('datasets/decision_tree_dataset.csv')
    print(data.head())
    X = data.iloc[:,1:2].astype(int).values
    y = data.iloc[:,2].astype(int).values.reshape(-1,1)


    regressor = RandomForestRegressor(n_estimators=5, min_samples_split=2, max_depth=4, random_state=0)
    regressor.fit(X,y.ravel()) 
    
    # My Code
    forest_regress = random_forest_regessor(num_trees=5, min_samples_split=2, max_depth=4)
    forest_regress.fit(X,y)

    X_grid = np.arange(min(X), max(X), 0.01)
    # converts to a column vector
    X_grid = X_grid.reshape((len(X_grid), 1)) 

    percent = 100 * (X.shape[0] - np.count_nonzero(forest_regress.predict(X) - regressor.predict(X))) / X.shape[0]

    plt.scatter(X,y, color='blue')
    plt.plot(X_grid, forest_regress.predict(X_grid), color='green', label='My Graph')
    plt.plot(X_grid, regressor.predict(X_grid), color='orange', label='Scitkit_Learn Graph')
    plt.title("Profit to Production Cost - Decision Tree Regression")
    plt.xlabel('Production Cost')
    plt.ylabel('Profit')
    plt.text(max(X)*.6, 0, f"Similiar = {percent: 0.2f}%", fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()