import numpy as np
import pandas as pd
import os
import argparse
import sklearn.metrics as metrics

from utils.dataset import AI4VN_AirDataset
from utils.models import EnsembleModel
from utils.evaluation import eval_regression_model


def get_args():
    parser = argparse.ArgumentParser()
    pass

if __name__ == "__main__":
    air_data = AI4VN_AirDataset(drop_null=True)
    X_train, X_test, y_train, y_test = air_data.get_data_loader()
    ensemble_model = EnsembleModel()
    trained_model = ensemble_model.fit(X_train, y_train)
    eval_regression_model(trained_model, X_test, y_test)

