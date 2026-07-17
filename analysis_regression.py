# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import sklearn
import helpers_regression as h

# Load the dataset
raw_data = pd.read_csv("auto-mpg.csv")

# Drop the car_name column
raw_data = raw_data.drop(columns=['car_name'])

# Drop the 'origin' column (don't think one-hot encoding adds much value)
raw_data = raw_data.drop(columns=['origin'])

# Randomize the row order
raw_data = raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels using column names first, then convert to NumPy arrays
X_raw = raw_data.drop(columns=['mpg']).to_numpy()
y_raw = raw_data['mpg'].to_numpy()

#################
# MODEL TESTING #
#################
# Two methods: 80/20 split and k-fold cross-validation

# 80/20 split
print('Training/testing regression and ridge regression models using 80/20 split:')
print('Splitting and standardizing the data...')
split_idx = int(len(X_raw) * 0.8)

X_raw_training = X_raw[:split_idx]
y_raw_training = y_raw[:split_idx]
X_raw_testing = X_raw[split_idx:]
y_raw_testing = y_raw[split_idx:]

# Standardize each set using training set statistics to prevent data leakage
X_train_mean = X_raw_training.mean(axis=0)
X_train_std = X_raw_training.std(axis=0)
y_train_mean = y_raw_training.mean()
y_train_std = y_raw_training.std()

X_train = (X_raw_training - X_train_mean) / X_train_std
X_test = (X_raw_testing - X_train_mean) / X_train_std

y_train = (y_raw_training - y_train_mean) / y_train_std
y_test = (y_raw_testing - y_train_mean) / y_train_std

# Add a column of 1s to the standardized matrices (bias trick)
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# Train regression and ridge regression models
print('Training the models...')
initial_theta = np.random.randn(X_train.shape[1])
r_theta_history, r_obj_history = h.grad_descent(X_train, y_train, 0, 0.01, 1000, initial_theta=initial_theta)
rr_theta_history, rr_obj_history = h.grad_descent(X_train, y_train, 0.01, 0.01, 1000, initial_theta=initial_theta)

regression_theta = r_theta_history[-1]
ridge_theta = rr_theta_history[-1]

# Use mean square error as evaluation metric
print('Using mean square error as evaluation metric:')
regression_mse = h.regression_objective(X_test, y_test, regression_theta)
ridge_mse = h.regression_objective(X_test, y_test, ridge_theta)
print('Regression MSE:', regression_mse)
print('Ridge Regression MSE:', ridge_mse)

# k-fold cross-validation
print('\nTraining/testing regression and ridge regression models using k-fold cross-validation:')
print('This may take a second...')
k = 10

X_folds = np.array_split(X_raw, k)
y_folds = np.array_split(y_raw, k)

r_mses = []
rr_mses = []

for i in range(k):
    # Split into train and validation folds
    X_train_raw = np.concatenate([X_folds[j] for j in range(k) if j != i])
    y_train_raw = np.concatenate([y_folds[j] for j in range(k) if j != i])
    X_val_raw = X_folds[i]
    y_val_raw = y_folds[i]
    
    # Standardize using training statistics
    X_train_mean = X_train_raw.mean(axis=0)
    X_train_std = X_train_raw.std(axis=0)
    y_train_mean = y_train_raw.mean()
    y_train_std = y_train_raw.std()
    
    X_train = (X_train_raw - X_train_mean) / X_train_std
    X_val = (X_val_raw - X_train_mean) / X_train_std
    y_train = (y_train_raw - y_train_mean) / y_train_std
    y_val = (y_val_raw - y_train_mean) / y_train_std
    
    # Add bias column
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
    # Train
    initial_theta = np.random.randn(X_train.shape[1])
    r_theta_history, _ = h.grad_descent(X_train, y_train, 0, 0.01, 1000, initial_theta=initial_theta)
    rr_theta_history, _ = h.grad_descent(X_train, y_train, 0.01, 0.01, 1000, initial_theta=initial_theta)
    
    # Evaluate
    r_mses.append(h.regression_objective(X_val, y_val, r_theta_history[-1]))
    rr_mses.append(h.regression_objective(X_val, y_val, rr_theta_history[-1]))

print('Average Regression MSE:', np.mean(r_mses))
print('Average Ridge Regression MSE:', np.mean(rr_mses))

########################################
# COMPARING AGAINST ANALYTICAL RESULTS #
########################################
# Full dataset standardization for the analytical baseline
X_full = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
X_full = np.hstack([X_full, np.ones((X_full.shape[0], 1))])
y_full = (y_raw - y_raw.mean()) / y_raw.std()

# Train analytical models on the full dataset (theoretical minimum)
reg_analyt = h.analytical_regression(X_full, y_full)
ridge_analyt = h.analytical_ridge_regression(X_full, y_full, 0.01)

# Evaluate analytical models on the full dataset
analyt_mse = h.regression_objective(X_full, y_full, reg_analyt)
ridge_analyt_mse = h.regression_objective(X_full, y_full, ridge_analyt)

print('\nAnalytical regression MSE:', analyt_mse)
print('Analytical ridge regression MSE:', ridge_analyt_mse)

