import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    np.savetxt(pred_path, clf.predict(x_test) > 0.5, fmt="%i")
    util.plot(x_test, y_test, clf.theta, pred_path + ".png")
    util.plot(x_train, y_train, clf.theta, pred_path + "_train.png")
    #
    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = 1 / m * np.sum(y[y == 1])
        mu_0 = np.sum(x * (y == 0)[:, None], axis=0) / np.sum((y == 0))
        mu_1 = np.sum(x * (y == 1)[:, None], axis=0) / np.sum((y == 1))
        _mu_0, _mu1_1 = mu_0 * (y == 0)[:, None], mu_1 * (y == 1)[:, None]
        sigma = 1 / m * (x - _mu_0 - _mu1_1).T @ (x - _mu_0 - _mu1_1)
        sigma_inv = np.linalg.inv(sigma)
        self.theta = np.zeros((n + 1,))
        self.theta[0] = 1 / 2 * (mu_0 - mu_1).T @ sigma_inv @ (mu_1 + mu_0) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv @ (mu_1 - mu_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(- (x @ self.theta)))
        # *** END CODE HERE
