# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import sklearn

##########################
# OBJECTIVE AND GRADIENT #
##########################
def regression_objective(X : np.ndarray, y : np.ndarray, theta : np.ndarray) -> float:
    """Objective function for linear regression using only mean squared error.
    - X is your data matrix (row vectors)
    - y is your target variable (column vector)
    - theta is the hyperplane normal column vector (bias is the last entry)"""
    n = X.shape[0]
    
    error = X @ theta - y

    return (1 / n) * error @ error

def regression_gradient(X : np.ndarray, y : np.ndarray, theta : np.ndarray) -> np.ndarray:
    """Gradient of the objective function for linear regression."""
    n = X.shape[0]

    error = X @ theta - y
    return (2 / n) * X.T @ error

def ridge_regression_objective(X : np.ndarray, y : np.ndarray, theta : np.ndarray, lam : float) -> float:
    """Objective function for ridge regression using MSE with regularization."""
    n = X.shape[0]
    
    error = X @ theta - y

    return (1 / n) * error @ error + lam * (theta[:-1] @ theta[:-1])

def ridge_regression_gradient(X : np.ndarray, y : np.ndarray, theta : np.ndarray, lam : float) -> np.ndarray:
    """Gradient of the objective function for ridge regression."""
    n = X.shape[0]

    reg_theta = np.copy(theta)
    reg_theta[-1] = 0.0

    error = X @ theta - y
    return (2 / n) * X.T @ error + 2 * lam * reg_theta

####################
# GRADIENT DESCENT #
####################
def grad_step(theta : np.ndarray, X : np.ndarray, y : np.ndarray, lam : float, alpha : float) -> tuple[np.ndarray, float]:
    """Takes weights, X, y, lam, learning rate, and returns weights 
    and objective after one step of gradient descent"""
    theta = theta - alpha * ridge_regression_gradient(X, y, theta, lam)
    return theta, ridge_regression_objective(X, y, theta, lam)

def grad_descent(X : np.ndarray, y : np.ndarray, lam : float, alpha : float, iterations : int, initial_theta : np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Takes X, y, lambda, learning rate, and max iterations, and returns a matrix of weights
    (row vectors) and array of objective values from each iteration of gradient descent."""
    # theta init
    if initial_theta is None:
        theta = np.zeros(X.shape[1])
    else:
        if initial_theta.shape[0] != X.shape[1]:
            raise ValueError("Initial theta must have the same dimension as X.")
        theta = np.array(initial_theta, dtype=float)

    # list to store objective values
    obj_values = [ridge_regression_objective(X, y, theta, lam)]

    # list to store theta history
    theta_history = [theta]

    for i in range(iterations):
        theta_new, obj_new = grad_step(theta_history[-1], X, y, lam, alpha)
        theta_history.append(theta_new)
        obj_values.append(obj_new)

    return np.array(theta_history), np.array(obj_values)

########################
# ANALYTICAL SOLUTIONS #
########################
def analytical_regression(X : np.ndarray, y : np.ndarray) -> np.ndarray:
    """Analytical solution for linear regression."""
    cov = X.T @ X
    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        raise ValueError("Covariance matrix is not invertible :(")
    return np.linalg.inv(cov) @ X.T @ y

def analytical_ridge_regression(X : np.ndarray, y : np.ndarray, lam : float) -> np.ndarray:
    """Analytical solution for ridge regression. Don't input negative values for lam."""
    if lam < 0:
        raise ValueError("Lambda cannot be negative.")
    if lam == 0:
        return analytical_regression(X, y)
    else:
        n = X.shape[0]
        I = np.eye(X.shape[1])
        I[-1, -1] = 0
        return np.linalg.inv(X.T @ X + n * lam * I) @ X.T @ y


