import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from sklearn.metrics import log_loss
from visualization.visualize import importance_plotting


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


    def calc_logreg_model(X_train, y_train):
        """
        
        Arguments:
            X_train {[type]} -- [description]
            y_train {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        print('feature list ...')
        print(f'{X_train.columns.tolist()}\n')
        
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train) 
    
        y_pred = pd.Series(logreg.predict(X_test), 
                           index=y_test.index, name='survived_pred').to_frame()
    
        scores = cross_val_score(logreg, X_train, y_train, cv=10)
        print(f"Cross Validation Accuracy Scores: {scores}")
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        
        return logreg, y_pred