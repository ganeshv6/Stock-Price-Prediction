from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return {"RMSE": rmse, "MAE": mae}
