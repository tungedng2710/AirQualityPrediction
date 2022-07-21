import numpy as np
import sklearn.metrics as metrics

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true + 1 - y_pred) / (y_true + 1)) * 100)

def eval_regression_model(model, X_test, y_test):
    y_true = y_test
    y_pred = model.predict(X_test)

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('R^2: ', r2)
    print('MAE: ', mean_absolute_error)
    print('MSE: ', mse)
    print('RMSE: ', np.sqrt(mse))
    print('MAPE: ', mean_absolute_percentage_error(y_true, y_pred), '%')