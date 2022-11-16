import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    y_preds = clf.predict(x_test)
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_preds, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("output/p05b_lwr")
    mse = np.mean((y_preds - y_test)**2)
    print("MSE = {}".format(np.round(mse, 4)))
    # *** END CODE HERE ***



class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_preds = np.zeros(m)
        for i in range(m):
            W = np.diag( np.exp(- np.linalg.norm(self.x - x[i], ord=2, axis=1)**2 / (2 * self.tau**2)))
            self.theta = np.linalg.inv(self.x.T @ W @ self.x) @ (self.x.T @ W @ self.y)
            y_preds[i] = self.theta.T @ x[i]
        return y_preds
        # *** END CODE HERE ***
