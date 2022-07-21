import sklearn
from sklearn import ensemble
import torch
import torch.nn as nn
import pickle
import os

class EnsembleModel:
    def __init__(self,
                 name: str = "gradientboosting"):
        self.name = name
        gbparams = {
                    "n_estimators": 500,
                    "max_depth": 4,
                    "min_samples_split": 5,
                    "learning_rate": 0.01,
                    "loss": "squared_error",
                   } 
        self.gradientboosting = ensemble.GradientBoostingRegressor(**gbparams)
        self.randomforest = ensemble.RandomForestRegressor(max_depth=2, random_state=0)
        if self.name == "gradientboosting":
            print("Creating Gradient Boosting Regressor")
            self.model = self.gradientboosting
        elif self.name == "randomforest":
            print("Creating Random Forest Regressor")
            self.model = self.randomforest
    
    def fit(self, 
            X_train = None, 
            y_train = None, 
            save_checkpoint = True,
            save_dir = "trained_model/"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print("Training model...")
        self.model.fit(X_train, y_train)
        if save_checkpoint:
            with open(save_dir+"EnsembleModel.pkl","wb") as f:
                pickle.dump(self.model,f)
                print("Trained model has been saved at "+save_dir+"EnsembleModel.pkl")
        return self.model

    def eval(self, X_test, y_test):
        pass

class NeuralNetwork:
    def __init__(self) -> None:
        super().__init__()
        pass
