#!/usr/bin/env python

import pandas as pd
import numpy as np
import datetime
from IPython.display import display, Markdown
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import cross_val_score, cross_val_score, cross_validate


def concat_to_create_xy_test(
    X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.array
):
    """Join X,y, and y_predictions to a common dataframe.  The index of the 
       all the dataframes must be the same. 
    
    Arguments:
        X_test [dataframe] -- Test data that matches your model.
        y_test [dataframe] -- Labels that match your model.
        y_pred [dataframe] -- Predicted labels.   
    
    Returns:
        [dataframe] -- Joined dataframe.  The column is_prediction_correct is
        added according to 

        Xy_test["is_prediction_correct"] = Xy_test["survived_pred"] == Xy_test["survived"]

    """
    Xy_test = X_test.join(y_test).join(y_pred)
    Xy_test["is_prediction_correct"] = Xy_test["survived_pred"] == Xy_test["survived"]

    return Xy_test


def calc_metrics(Xy_test):
    """
    Calculates the LogLoss and accuracy of the 
    
    Arguments:
        Xy_test [dataframe] -- This is the joined dataframe from created by 
        concat_to_create_xy_test
    
    Returns:
        dictionary -- Calculates the LogLoss and accuracy

        metric['log_loss']
        metric['accuracy']
    """
    metric = {}
    metric["log_loss"] = log_loss(
        Xy_test["survived"].values, Xy_test["survived_pred"].values
    )
    metric["accuracy"] = Xy_test["is_prediction_correct"].mean()

    return metric


def fit_and_predict_model(model, X_train, y_train, X_test, y_test):
    """
        Fit and predict the model.

        Arguments:
            X_train {array | dataframe} -- Training data to be fitted to the model. 
            y_train {array | dataframe} -- Training labels to be fitted to the model. 

        Returns:
            y_pred -- 
        """
    print("feature list ...")
    print(f"{X_train.columns.tolist()}\n")

    model.fit(X_train, y_train)

    y_pred = pd.Series(
        model.predict(X_test), index=y_test.index, name="survived_pred"
    ).to_frame()

    scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f"Cross Validation Accuracy Scores: {scores}")
    print("\n\nAccuracy: %0.4f (+/- %0.5f)\n\n" % (scores.mean(), scores.std() * 2))

    return y_pred


def calc_logreg_model(X_train, y_train, X_test, y_test):
    """

        Arguments:
            X_train {array | dataframe} -- Training data to be fitted to the model. 
            y_train {array | dataframe} -- Training labels to be fitted to the model. 

        Returns:
            logreg -- 
            y_pred -- [description]
        """
    print("feature list ...")
    print(f"{X_train.columns.tolist()}\n")

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = pd.Series(
        logreg.predict(X_test), index=y_test.index, name="survived_pred"
    ).to_frame()

    scores = cross_val_score(logreg, X_train, y_train, cv=10)
    print(f"Cross Validation Accuracy Scores: {scores}")
    print("\n\nAccuracy: %0.4f (+/- %0.5f)\n\n" % (scores.mean(), scores.std() * 2))

    return logreg, y_pred


def calc_model_predictions(model, X_test=None, y_test=None, verbose=True):
    """
    Calculate model predictions and display accuracy if verbose=True.
    
    Keyword Arguments:
        model {scikit_learn model} -- [description]
        X_test {[type]} -- [description] (default: {None})
        y_test {[type]} -- [description] (default: {None})
        verbose {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    if (X_test is not None) and (y_test is not None):

        y_pred = pd.Series(
            model.predict(X_test), index=y_test.index, name="survived_pred"
        ).to_frame()

        predicted_accuracy_score = metrics.accuracy_score(y_test, y_pred)

        if verbose:
            print(
                f"\nAccuracy Score on X_test,y_test: {predicted_accuracy_score: 0.4f}\n"
            )

    else:
        y_pred = None
        predicted_accuracy_score = None

    return y_pred, predicted_accuracy_score


def calc_model_rst_table_metrics(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    cv=5,
    model_name="<model>",
    scoring=("accuracy", "recall", "precision", "f1"),
    verbose=True,
):
    """
    Calculate the cross validation scores and test predictions. 

    Arguments:
        model {scikit_learn model} -- [description]
        X_train {dataframe} -- [description]
        y_train {dataframe} -- [description]

    Keyword Arguments:
        X_test {dataframe} -- [description] (default: {None})
        y_test {dataframe} -- [description] (default: {None})
        cv {int} -- [description] (default: {5})
        model_name {str} -- [description] (default: {"<model>"})
        scoring {tuple} -- [description] (default: {("accuracy", "recall", "precision", "f1")})
        verbose {bool} -- [description] (default: {True})

    Returns:
        y_pred -- Model classification prediction
        cv_scores - Cross validation scores from scoring tuple.
    """
    cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    y_pred, predicted_accuracy_score = calc_model_predictions(
        model, X_test, y_test, verbose=verbose
    )

    if verbose:
        display_rst_table_metrics_log(cv_scores, model_name=model_name)

    return y_pred, predicted_accuracy_score, cv_scores


def display_rst_table_metrics_log(cv_scores, model_name="<model_name>"):
    """Display RST Table Metric Logs for a single cross validation run. 

    Here is an example of the output.  This function produces a single line of the output
    for tracking.  If one wishes to track the results they may appended to the
    model_accuracy.csv file.  

    | date,     model,                holdout_accuracy, accuracy, recall,     precision, f1
    | 10/28/19, dummy_most_frequent,  NS,               0.6223,         ,           ,           
    | 10/28/19, logreg_gender_only,   0.76555,          0.7958,   0.6959,     0.7455,     0.7189
    | 10/28/19, logreg model_1,       0.77990,          0.8286,   0.7238,     0.8013,     0.7602 
    | 10/28/19, logreg model_2,       0.78468,          0.8286,   0.7200,     0.8037,     0.7592 
    | 10/28/19, logreg model_3,       0.79425,          0.8384,   0.7162,     0.8306,     0.7691
    | 11/05/19, dtree_model_1,        0.77990,          0.8173,   0.6486,     0.8310,     0.7251
    | 11/05/19, dtree_model_2,        0.78468,          0.8272,   0.7012,     0.8143,     0.7516


    Arguments:
        cv_scores {List} -- The CV scores calculated calc_model_rst_table_metrics.
    
    Returns:
         None
    """

    print("\nCross Validation Scores:")
    print(
        "\tAccuracy \t: %0.4f (+/- %0.4f)"
        % (cv_scores["test_accuracy"].mean(), cv_scores["test_accuracy"].std() * 2)
    )
    print(
        "\tRecall\t\t: %0.4f (+/- %0.4f)"
        % (cv_scores["test_recall"].mean(), cv_scores["test_recall"].std() * 2)
    )
    print(
        "\tPrecision\t: %0.4f (+/- %0.4f)"
        % (cv_scores["test_precision"].mean(), cv_scores["test_precision"].std() * 2)
    )
    print(
        "\tF1\t\t: %0.4f (+/- %0.4f)"
        % (cv_scores["test_f1"].mean(), cv_scores["test_f1"].std() * 2)
    )

    print(
        f"\n\n{datetime.datetime.now().strftime('%m/%d/%y')}, {model_name},  <kaggle_accuracy>, {cv_scores['test_accuracy'].mean():0.4f}, {cv_scores['test_recall'].mean():0.4f},{cv_scores['test_precision'].mean():0.4f},{cv_scores['test_f1'].mean():0.4f}"
    )
