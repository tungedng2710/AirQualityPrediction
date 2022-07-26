'''
Script running is in progress
Please check the Quickstart notebook for instant use
'''

from utils.dataset import AI4VN_AirDataLoader
from utils.evaluation import eval_regression_model
from utils.models import EnsembleModel

if __name__ == "__main__":
    air_data_loader = AI4VN_AirDataLoader(test_size=0.15)
    X_train, X_test, y_train, y_test = air_data_loader.get_data_loader_sklearn()
    ensemble_model = EnsembleModel(name='randomforest')
    trained_model = ensemble_model.fit(X_train, y_train)
    eval_regression_model(trained_model, X_test, y_test)

