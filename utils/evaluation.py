import numpy as np
import sklearn.metrics as metrics


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true + 1 - y_pred) / (y_true + 1)) * 100)


def median_absolute_percentage_error(y_true, y_pred):
    return np.median((np.abs(np.subtract(y_true, y_pred) / y_true))) * 100


def eval_regression_model(y_true, y_pred, verbose=1):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    if verbose > 0:
        print('explained_variance: ', round(explained_variance, 4))
        print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
        print('R^2: ', round(r2, 4))
        print('MAE: ', round(mae, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(rmse, 4))
        print('MAPE: ', round(mape, 4))
        print('MDAPE: ', round(mdape, 4))
    return r2, mae, mse, rmse, mape, mdape
