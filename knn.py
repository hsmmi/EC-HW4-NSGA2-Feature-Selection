from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class knn:
    def __init__(self, k=3, n_fold=3):
        """
        Initialize k-NN classifier
        """
        self.k = k
        self.n_fold = n_fold

    def fit(self, X, y):
        """
        Fit k-NN classifier
        """
        # Store training data
        self.X = X
        self.y = y
        # Get number of samples and features
        self.n_samples, self.n_features = X.shape

    def accuracy(self, solution):
        """
        Evaluate solution using k-NN (k = 3) with n fold cross-validation (n = 3)
        """
        # Split data into n folds
        folds = np.array_split(np.random.permutation(self.n_samples), self.n_fold)
        # Evaluate solution using n fold cross-validation
        error = 0
        for i in range(self.n_fold):
            # Get train and test data
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_fold) if j != i])
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_test, y_test = self.X[test_idx], self.y[test_idx]
            # Evaluate solution
            error += self.knn(X_train, y_train, X_test, y_test, solution)
        # Return average error
        return error / self.n_fold

    def knn(self, X_train, y_train, X_test, y_test, solution):
        """
        k-NN (k = 3)
        """
        # Get selected features
        selected_features = np.where(solution == 1)[0]
        # If no feature is selected, return 1
        if len(selected_features) == 0:
            return 1
        # Get selected features
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        # Fit k-NN classifier
        clf = KNeighborsClassifier(n_neighbors=self.k)
        clf.fit(X_train, y_train)
        # Predict
        y_pred = clf.predict(X_test)
        # Return error
        return 1 - accuracy_score(y_test, y_pred)
