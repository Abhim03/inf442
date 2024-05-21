import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split


def plot_points_and_decision_boundary(X: np.array, y: np.array, clf):
    """
    Given the datapoints (X), class labels (y), and a fitted perceptron (clf),
    it plots:
      - the datapoints, coloring them based on the class label
      - the decision boundary of the classifier
    """
    h = 0.02
    x_min, x_max = X[:,0].min() - 10 * h, X[:,0].max() + 10 * h
    y_min, y_max = X[:,1].min() - 10 * h, X[:,1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap='Paired_r', alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='Paired_r', edgecolors='k')
    plt.show()


def make_checker(n_samples: int):
    """
    Makes a checker-shaped dataset, return X and y
    """
    # setting the seed to get the same result at each run
    np.random.seed(42)
    points = np.random.uniform(size=(n_samples, 2))
    classes = np.ones(n_samples)
    quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    class_label = [1, -1, 1, -1]
    for i in range(n_samples):
        q = np.random.randint(0, 4)
        points[i, 0] = points[i, 0] * quadrants[q][0]
        points[i, 1] = points[i, 1] * quadrants[q][1]
        classes[i] = class_label[q]
    return points, classes


# Two-circles dataset
X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=1)

# Two-moon dataset
#X, y = make_moons(n_samples=500, noise=0.02, random_state=1)

# Checker dataset
# X, y = make_checker(500)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# In the quiz, you may want to change `hidden_layer_sizes` and `activation` 
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(2, 2), activation='relu', learning_rate_init=0.1, tol=1e-6, max_iter=10000)


clf.fit(X_train, y_train)

plot_points_and_decision_boundary(X, y, clf)

print(f"Accuracy on the training set: {clf.score(X_train, y_train)}")
print(f"Accuracy in the test set: {clf.score(X_test, y_test)}")

