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


def rank_features_by_logreg_coefficients(X_train, logreg, abs_threshold = 0.5):
    
    fi = {'Features':X_train.columns.tolist(), 'Importance':np.transpose(logreg.coef_[0])}

    importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False).set_index('Features').sort_values(by='Importance', ascending=False)

    mask = abs(importance['Importance'])>abs_threshold

    with pd.option_context('display.max_rows', len(importance)):
        display(importance[mask])
    
    # Creating graph title
    titles = ['The most important features in predicting survival on the Titanic: Logistic Regression']

    # Plotting graph
    importance_plotting(importance.reset_index(), 'Importance', 'Features', 'Reds_r', titles)
    
    important_features = importance[mask].index.tolist()
    unimportant_features = importance[~mask].index.tolist()
    return important_features, unimportant_features