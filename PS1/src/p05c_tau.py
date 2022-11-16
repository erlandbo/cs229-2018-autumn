import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    mses = {}
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        plt.plot(x_train, y_train, 'bx')
        plt.plot(x_val, y_preds, 'ro')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("tau = {}".format(tau))
        plt.savefig("output/p05c_tau_{}.png".format(tau), format='png')
        plt.close()
        mse = np.mean((y_preds-y_val)**2)
        mses[tau] = mse
        print("tau={} mse ={}".format(tau, mse.round(4)))
    tau = sorted(mses, key=mses.get)[0]
    model = LocallyWeightedLinearRegression(tau=tau)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    mse_test = np.mean((y_preds - y_test)**2)
    print("TEST: tau={} mse ={}".format(tau, np.round(mse_test, 4)))
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_preds, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("output/p05c_test_tau_{}.png".format(tau), format='png')
    np.savetxt(pred_path, y_preds)
    # *** END CODE HERE ***
