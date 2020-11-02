import matplotlib.pyplot as plt
import numpy as np
import sklearn
from src.linear_model import LinearRegression
from src.ml_tools import split_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)

    n = 100
    x = np.random.randn(n, 1)
    noise = np.random.randn(n, 1)
    y = 2 * x + noise

    x_train, y_train, x_test, y_test = split_data(x, y, test_ratio=0.2)

    linreg = LinearRegression()
    linreg.fit(x, y)

    y_tilde_train = linreg.predict(x_train)
    y_tilde_test = linreg.predict(x_test)

    print("Train MSE:", linreg.mse(x_train, y_train))
    print("Train R2:", linreg.R2(x_train, y_train))
    print("Test MSE:", linreg.mse(x_test, y_test))
    print("Test R2:", linreg.R2(x_test, y_test))

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, ytilde_train, "r", label="Train data")
    plt.plot(x, y_tilde_test, "k", label="Test data")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.show()
