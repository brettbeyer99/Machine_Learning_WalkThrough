from sklearn.datasets import load_iris
import numpy as np

class DecisionTreeClassifier():
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def calc_criterion(self, y):
        pass

    def fit(self, X, y):
        self.tree = self.build_tree(X,y)

    def build_tree(self, X, y, depth=0):
        rows, cols = X.shape
        unique_classes = np.unique(y)
        print(unique_classes)

        # stopping conditions
        if len(unique_classes)==1 or depth==self.max_depth or rows<self.min_samples_split:
            leaf_value = self.calculate_leaf_value(y)
            return {"leaf": leaf_value}
        
        feature_index, threshold, gain = self.best_split(X, y)

        if gain==0:
            leaf_value = self.calculate_leaf_value(y)
            return {"leaf": leaf_value}
        
        X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
        left_branch = self.build_tree(X_left, y_left, depth+1)
        right_branch = self.build_tree(X_right, y_right, depth+1)

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left_branch": left_branch,
            "right_branch": right_branch
        }
        

    def calculate_leaf_value(self, y):
        print("Leaf Value = " + str(np.bincount(y).argmax()))
        return np.bincount(y).argmax()   

    def best_split(self, X, y):
        best_gain = -float("inf")
        best_feature, best_threshold = None, None
        cols = X.shape[1]

        for feature_index in range(cols):
            # print("FEATURE INDICES: ")
            # print(X[:,feature_index])
            thresholds = np.unique(X[:,feature_index])
            # print("Thresholds = " + str(thresholds))
            for threshold in thresholds:
                # print("Theshold Value = " + str(threshold))
                X_left, X_right, y_left, y_right = self.split(X,y,feature_index, threshold)
                # print("Threshold Indices: ")
                # print(X_left)
                # print(X_right)
                # print(y_left)
                # print(y_right)
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = self.calculate_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_index
                        best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def split(self, X, y, feature_index, threshold):
        # find the indices where values in feature_index are less than/equal to or greater than threshold
        left_index = np.where(X[:,feature_index] <= threshold)
        right_index = np.where(X[:,feature_index] > threshold)
        return X[left_index], X[right_index], y[left_index], y[right_index]

    def calculate_gain(self, y, y_left, y_right):
        # first calculate weights
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        gain = self.calculate_criterion(y) - (weight_left*self.calculate_criterion(y_left) + weight_right*self.calculate_criterion(y_right))
        return gain

    def calculate_criterion(self, y):
        unique_y, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == 'gini':
            return 1 - np.sum(probabilities**2)
        elif self.criterion == 'entropy':
            pass
        else:
            raise ValueError("Invalid Criterion Entered: Please choose 'gini' or 'entropy'.")

if __name__ == "__main__":
    
    # Use sklearn iris dataset
    data = load_iris()
    X, y = data.data, data.target
    # print(X)
    # print(X[1])
    # print(y)

    dc_classifier = DecisionTreeClassifier()
    dc_classifier.fit(X,y)



