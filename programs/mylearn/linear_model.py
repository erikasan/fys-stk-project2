#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class StatisticalMetrics:

    def rss(self, data, target):
        """
        RSS - Residual Sum of Squares
        """
        return np.sum((target - self.predict(data))**2)

    def sst(self, target):
        """
        SST - Sum of Squares Total
        """
        return np.sum((target - np.mean(target))**2)

    def r2(self, data, target):
        """
        Calculate the R^2-score, coefficient of determination (R^2-score)
        """
        return 1 - self.rss(data, target) / self.sst(target)

    def mse(self, data, target):
        """
        MSE - Mean Squared Error
        """
        return np.mean((target - self.predict(data))**2)


class LinearRegression(StatisticalMetrics):
    """
    Ordinary Least Squares (OLS) Regression
    """

    # the intercept, adds 1 with True.
    def __init__(self, fit_intercept=True, normalize=False):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
        self._normalize = normalize

    def normalize_data(self):
        """
        Normalize data with the exception of the intercept column
        """
        if self._fit_intercept:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)

            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return data_norm
        else:
            self.data_mean = np.mean(self.data[:, 1:], axis=0)
            self.data_std = np.std(self.data[:, 1:], axis=0)
            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return np.c_[np.ones(X.shape[0]), data_norm]

    def fit(self, X, y):
        """
        Fit the model
        ----------
        Input: design matrix (data), target data
        """
        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:  # find shape of array
            # reshape takes all data with [-1] and makes it 2D
            _X = self.data.reshape(-1, 1)
        else:
            _X = self.data

        # if normalize data
        if self._normalize:
            _X = self.normalize_data()

        # add bias if fit_intercept
        if self._fit_intercept:
            _X = np.c_[np.ones(X.shape[0]), _X]

        self._inv_xTx = np.linalg.pinv(_X.T @ _X)  # pseudo-inverse
        beta = self._inv_xTx @ _X.T @ self.target

        # set attributes
        if self._fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = np.mean(self.target)
            self.coef_ = beta

        return self.coef_

    def predict(self, X):
        """
        Model prediction
        """
        # reshapes if X is wrong
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_

    def coef_CI(self, critical_value=1.96):
        """
        Estimate a confidence interval of the coefficients

        The critical value for a 90% confidence interval is 1.645
        The critical value for a 95% confidence interval is 1.96
        The critical value for a 98% confidence interval is 2.326
        The critical value for a 99% confidence interval is 2.576

        Returns lower and upper bound as sets in a list.
        """
        beta_std = np.sqrt(np.diag(self._inv_xTx))
        beta = self.coef_
        data_mse_sqrt = np.sqrt(self.mse(self.data, self.target))
        CI = [[beta[i] - critical_value * beta_std[i] * data_mse_sqrt, beta[i] +
               critical_value * beta_std[i] * data_mse_sqrt]for i in range(len(beta))]
        return CI


class RidgeRegression(StatisticalMetrics):
    """
    Linear Model Using Ridge Regression.
    """

    def __init__(self, lmbda=1.0, fit_intercept=True, normalize=False):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept
        self._normalize = normalize

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        if isintstance(value, (int, float)):
            self._lmbda = value
        else:
            raise ValueError("Penalty must be int or float")

    def normalize_data(self):
        """
        Normalize data with the exception of the intercept column
        """
        if self._fit_intercept:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)

            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return data_norm
        else:
            self.data_mean = np.mean(self.data[:, 1:], axis=0)
            self.data_std = np.std(self.data[:, 1:], axis=0)
            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return np.c_[np.ones(X.shape[0]), data_norm]

    def fit(self, X, y):

        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:  # find shape of array
            # reshape takes all data with [-1] and makes it 2D
            _X = self.data.reshape(-1, 1)
        else:
            _X = self.data

        # if normalize data
        if self._normalize:
            _X = self.normalize_data()

        # add bias if fit_intercept
        if self._fit_intercept:
            _X = np.c_[np.ones(X.shape[0]), _X]

        # calculate coefficients
        xTx = _X.T @ _X
        lmb_eye = self._lmbda * np.identity(xTx.shape[0])
        _inv_xTx = np.linalg.pinv(xTx + lmb_eye)  # pseudo-inverse
        coef = _inv_xTx @ _X.T @ self.target

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = np.mean(self.target)
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Model prediction
        """
        # reshapes if X is wrong
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_


class GDRegressor:
    """
    """

    def __init__(self, eta=0.01, n_iter=1000, tol=1e-5, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._eta = eta
        self._n_iter = n_iter
        self._tol = tol
        self._fit_intercept = fit_intercept

    def fit(self, X, y, weights=None, method="BGD"):
        """
        Fit the model
        ----------
        Input: design matrix (data), target data
        """
        self.data = X
        self.target = y
        self._method = method

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:  # find shape of array
            # reshape takes all data with [-1] and makes it 2D
            _X = self.data.reshape(-1, 1)
        else:
            _X = self.data

        """
        # if normalize data
        if self._normalize:
            _X = self.normalize_data()
        """

        # initialize coefficients
        if weights is not None:
            self.coef_ = weights
        else:
            if self._fit_intercept:
                self.coef_ = np.zeros(X.shape[1] + 1)
            else:
                self.coef_ = np.zeros(X.shape[1])

        # add bias if fit_intercept
        if self._fit_intercept:
            _X = np.c_[np.ones(X.shape[0]), _X]

        # compute coefficients
        if self._method == "Inv":
            coef = self._inversion(_X)

        elif self._method == "BGD":
            coef = self._batchGD(_X)

        elif self._method == "SGD":
            coef = self._stochasticGD(_X)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = np.mean(self.target)
            self.coef_ = coef

        return self.coef_

    def _inversion(self, _X):
        self._inv_xTx = np.linalg.pinv(_X.T @ _X)  # pseudo-inverse
        coef = self._inv_xTx @ _X.T @ self.target
        return coef

    def _batchGD(self, _X):

        m = _X.shape[0]
        factor = 2 / m
        coef = self.coef_

        for iteration in range(self._n_iter):
            coef_old = coef
            gradients = factor * _X.T @ (_X @ coef - self.target)
            coef = coef_old - self._eta * gradients
            dL2 = np.linalg.norm(coef - coef_old)
            if dL2 < self._tol or coef[0] != coef[0]:
                break
        return coef

    def _stochasticGD(self, _X):
        m = _X.shape[0]
        n_epochs = 50
        t0, t1 = 5, 50  # learning schedule hyperparameters
        coef = self.coef_

        def learning_schedule(t):
            """
            Determines the learning rate at each iteration
            """
            return t0 / (t + t1)

        # coef = np.random.randn(2, 1)  # random initialization

        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = _X[random_index:random_index + 1]
                yi = self.target[random_index:random_index + 1]
                gradients = 2 * xi.T @ ((xi @ coef) - yi)
                eta = learning_schedule(epoch * m + i)
                coef = coef - eta * gradients

        return coef

    def predict(self, X):
        """
        Prediction
        """
        # reshapes if X is wrong
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_
