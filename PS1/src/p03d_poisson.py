import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression(step_size=lr, max_iter=10000)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)
    plt.plot(y_pred, 'ro')
    plt.plot(y_test, 'bx')
    plt.savefig(pred_path+"_1_.png")
    plt.close()
    plt.plot(y_test, y_pred,'bx')
    plt.savefig(pred_path+"_2_.png")

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        alpha = self.step_size
        self.theta = np.zeros(n)
        theta_0 = np.random.rand(n)
        for _ in range(self.max_iter):
            h = np.exp(x @ self.theta)
            self.theta += alpha * x.T @ (y-h) * 1/m
            if np.linalg.norm(self.theta - theta_0, ord=1) < self.eps: break
            theta_0 = np.copy(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
