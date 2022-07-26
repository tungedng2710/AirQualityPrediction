from sklearn import ensemble
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

torch.manual_seed(42)


class EnsembleModel:
    def __init__(self,
                 name: str = "gradientboosting"):
        self.name = name
        gbparams = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "ls",
        }
        self.gradientboosting = ensemble.GradientBoostingRegressor(**gbparams)
        self.randomforest = ensemble.RandomForestRegressor(max_depth=2, random_state=0)
        if self.name == "gradientboosting":
            print("Creating Gradient Boosting Regressor")
            self.model = self.gradientboosting
            self.alias = "GB"
        elif self.name == "randomforest":
            print("Creating Random Forest Regressor")
            self.model = self.randomforest
            self.alias = "RF"

    def fit(self,
            X_train=None,
            y_train=None,
            save_checkpoint=True,
            save_dir="trained_models/"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print("Training model...")
        self.model.fit(X_train, y_train)
        if save_checkpoint:
            with open(save_dir + self.alias + "_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
                print("Trained model has been saved at " + save_dir + self.alias + "_model.pkl")
        return self.model


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalizedLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        return F.linear(x, W)


class NeuralNetwork(nn.Module):
    def __init__(self,
                 num_input_feat: int = 2):
        super(NeuralNetwork, self).__init__()
        linear1 = NormalizedLinear(num_input_feat, 64)
        relu = nn.ReLU()
        linear2 = NormalizedLinear(64, 32)
        linear3 = NormalizedLinear(32, 1)
        self.mlp = nn.Sequential(linear1, relu, linear2, relu, linear3)

    def forward(self, x):
        return self.mlp(x)
