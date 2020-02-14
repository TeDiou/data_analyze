import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

data = load_boston()
train_X, test_X, train_y, test_y = \
    train_test_split(data.data, data.target, test_size=0.25)
ada_model = AdaBoostRegressor(n_estimators=50, random_state=1)
ada_model.fit(train_X, train_y)
pred = ada_model.predict(test_X)
dt_model = DecisionTreeRegressor()
dt_model.fit(train_X, train_y)
dt_pred = dt_model.predict(test_X)

print(metrics.mean_absolute_error(test_y, pred))
print(metrics.mean_absolute_error(test_y, dt_pred))
