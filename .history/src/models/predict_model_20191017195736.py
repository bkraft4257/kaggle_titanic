import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


def concat_to_create_xy_test(X_test:pd.DataFrame, y_test: pd.DataFrame, y_pred: np.array):
    """[summary]
    
    Arguments:
        X_test {[dataframe]} -- [description]
        y_test {[dataframe]} -- [description]
        y_pred {[np.array]} -- [description]
    
    Returns:
        [dataframe] -- [description]
    """
    Xy_test = X_test.join(y_test).join(y_pred)
    Xy_test['is_prediction_correct'] = Xy_test['survived_pred'] == Xy_test['survived']
    
    return Xy_test


def calc_metrics(Xy_test):
    metric = {}
    metric['log_loss'] = log_loss(Xy_test['survived'].values, Xy_test['survived_pred'].values)
    metric['accuracy'] = Xy_test['is_prediction_correct'].mean()
    
    return metric