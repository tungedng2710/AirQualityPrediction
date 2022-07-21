import numpy as np
import pandas as pd
import os
import argparse
import sklearn.metrics as metrics

from dataset import AI4VN_AirDataset
from models import EnsembleModel


def get_args():
    parser = argparse.ArgumentParser()
    pass

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

if __name__ == "__main__":
    air_data = AI4VN_AirDataset(drop_null=True)
    X_train, X_test, y_train, y_test = air_data.get_data_loader()
    ensemble_model = EnsembleModel()
    trained_model = ensemble_model.fit(X_train, y_train)
    eval_regression_model(trained_model, X_test, y_test)

