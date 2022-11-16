import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    np.savetxt(pred_path, clf.predict(x_test) > 0.5, fmt="%i")
    util.plot(x_test, y_test, clf.theta, pred_path+".png")
    util.plot(x_train, y_train, clf.theta, pred_path+"_train.png")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = theta_0 = np.zeros((n,))
        g = lambda z: 1 / (1 + np.exp(-z))
        for _ in range(self.max_iter):
            B = np.zeros((m, m))
            np.fill_diagonal(B, g(x @ self.theta) * (1 - g(x @ self.theta)))
            grad = 1 / m * x.T @ (g(x @ self.theta) - y)
            H = 1 / m * x.T @ B @ x
            self.theta = self.theta - np.linalg.inv(H) @ grad
            if np.linalg.norm(self.theta - theta_0, 1) < self.eps: break
            theta_0 = self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda z: 1 / (1 + np.exp(-z))
        return g(x @ self.theta)
        # *** END CODE HERE ***
