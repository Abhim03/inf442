import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def train_and_evaluate(X_train, X_test, y_train, y_test, nneighbors, roc_fname="roc.png"):
    """
    Trains a kNN classifier and displays metrics of its performance on the test data;
    Returns the classifier
    """
    ## Training the classifier
    cls = KNeighborsClassifier(n_neighbors=nneighbors, algorithm='kd_tree')
    cls.fit(X_train, y_train)

    ## Predicting
    predictions = cls.predict(X_test)

    ## Computing the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix is:")
    print(cm)
    # if there are only two classes, compute more refined quantities
    # and print ROC curve
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"Error rate {(fp + fn) * 1. / (fp + fn + tp + tn)}")
        print(f"False alarm rate {fp * 1. / (fp + tn)}")
        detection = tp * 1. / (tp + fn)
        print(f"Detection rate {detection}")
        precision = tp * 1. / (tp + fp)
        print(f"Precision {precision}")
        print(f"F-score {2 * detection * precision / (detection + precision)}")

        ## Drawing a ROC curve
        RocCurveDisplay.from_estimator(cls, X_test, y_test)
        plt.savefig(roc_fname)

    return cls

# -------------------------------------------------------------------------------

def normalize(X_train, X_test, method='mean_std'):
    """
    Normalizes all the features by linear transformation.
    Two normalization methods are implemented:

      -- `mean_std` shifts by the mean and divides by the standard deviation

      -- `maxmin` shifts by the min and divides by the difference between max and min

      *Note*: mean/std/max/min are computed on the training data
    The function returns a pair normalized_train, normalized_test. For example,
    """
    # scaling
    normalized_train, normalized_test = None, None
    if method == 'mean_std':
        normalized_train = (X_train - X_train.mean()) / X_train.std()
        normalized_test = (X_test - X_train.mean()) / X_train.std()
    elif method == 'maxmin':
        normalized_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        normalized_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())
    else:
        raise f"Unknown method {method}"

    # gluing back the class column and returning
    return normalized_train, normalized_test

# -------------------------------------------------------------------------------

def get_audit_data():
    ## Reading the data
    train_fname = "../csv/audit_train.csv"
    test_fname = "../csv/audit_test.csv"
    col_class = 'Risk'
    train_data = pd.read_csv(train_fname, header=0)
    X_train, y_train = train_data.drop(col_class, axis=1), train_data[col_class]
    test_data = pd.read_csv(test_fname, header=0)
    X_test, y_test = test_data.drop(col_class, axis=1), test_data[col_class]

    return X_train, X_test, y_train, y_test

# -------------------------------------------------------------------------------

def get_digits_data():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

# -------------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_audit_data()

    ## Train & Test !
    N = 10
    cls = train_and_evaluate(X_train, X_test, y_train, y_test, N)
