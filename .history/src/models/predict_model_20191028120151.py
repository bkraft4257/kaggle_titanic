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
    """[summary]
    
    Arguments:
        X_test {[dataframe]} -- [description]
        y_test {[dataframe]} -- [description]
        y_pred {[np.array]} -- [description]
    
    Returns:
        [dataframe] -- [description]
    """
    Xy_test = X_test.join(y_test).join(y_pred)
    Xy_test["is_prediction_correct"] = Xy_test["survived_pred"] == Xy_test["survived"]

    return Xy_test


def calc_metrics(Xy_test):
    metric = {}
    metric["log_loss"] = log_loss(
        Xy_test["survived"].values, Xy_test["survived_pred"].values
    )
    metric["accuracy"] = Xy_test["is_prediction_correct"].mean()

    return metric


def calc_logreg_model(X_train, y_train, X_test, y_test):
    """

        Arguments:
            X_train {[type]} -- [description]
            y_train {[type]} -- [description]

        Returns:
            [type] -- [description]
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


def calc_model_rst_table_metrics(model, X_train, y_train):
    scores = cross_validate(
        model, X_train, y_train, cv=5, scoring=("accuracy", "recall", "precision", "f1")
    )

    print(
        "\n\nAccuracy: %0.4f (+/- %0.4f)"
        % (scores["test_accuracy"].mean(), scores["test_accuracy"].std() * 2)
    )
    print(
        "Recall: %0.4f (+/- %0.4f)"
        % (scores["test_recall"].mean(), scores["test_recall"].std() * 2)
    )
    print(
        "Precision: %0.4f (+/- %0.4f)"
        % (scores["test_precision"].mean(), scores["test_precision"].std() * 2)
    )
    print(
        "F1: %0.4f (+/- %0.4f)"
        % (scores["test_f1"].mean(), scores["test_f1"].std() * 2)
    )

    print(
        f"\n\n{datetime.datetime.now().strftime('%m/%d/')}, <model>, {scores['test_accuracy'].mean():0.4f}, {scores['test_recall'].mean():0.4f},{scores['test_precision'].mean():0.4f},{scores['test_f1'].mean():0.4f}, NS"
    )

