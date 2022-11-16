import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, t_train)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    np.savetxt(pred_path_c, clf.predict(x_test) > 0.5, fmt="%i")
    util.plot(x_test, t_test, clf.theta, pred_path_c + ".png")
    util.plot(x_train, t_train, clf.theta, pred_path_c + "train.png")

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf.fit(x_train, y_train)
    np.savetxt(pred_path_d, clf.predict(x_test) > 0.5, fmt="%i")
    util.plot(x_test, t_test, clf.theta, pred_path_d + ".png")
    util.plot(x_train, t_train, clf.theta, pred_path_d + "train.png")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    clf.fit(x_val, y_val)
    h = clf.predict(x_val)
    alpha = np.sum(h[y_val == 1]) / np.sum(y_val == 1)
    #clf.theta[0] += np.log(2/alpha - 1)
    np.savetxt(pred_path_e, clf.predict(x_test) /alpha> 0.5, fmt="%i")
    util.plot(x_test, t_test, clf.theta,pred_path_e + ".png", correction=alpha)
    util.plot(x_train, t_train, clf.theta, pred_path_e + "train.png", correction=alpha)

    # *** END CODER HERE
