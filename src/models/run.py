#!/usr/bin/env python

""" A command line interface for running models.


usage: run [-h] [-o OUTPUT_FILENAME] [-y Y] [-v] model X

positional arguments:
  model                 Filename of sklearn joblib model
  X                     Filename of feature data.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILENAME, --output_filename OUTPUT_FILENAME
                        Filename of submission to Kaggle
                        containingpredictions.
  -y Y                  Filename of known y data (if available). If not
                        available then metrics are not returned.
  -v, --verbose         Verbose flag



Returns:
    output file of y predictions in Kaggle submission format when requested (-o)
"""

import sys
import argparse
import pandas as pd
from joblib import load
from sklearn import metrics


def argparse_command_line():
    """  Extract parameters from the command line.

    
    Returns:
        {arg_parse object} -- Command line arguments as parsed with arg_parse.
    """    

    parser = argparse.ArgumentParser(prog="run")
    parser.add_argument("model", help="Filename of sklearn joblib model", type=str)
    parser.add_argument("X", help="Filename of feature data.", type=str)
    parser.add_argument(
        "-o",
        "--output_filename",
        help=("Filename of submission to Kaggle containing" "predictions."),
        type=str,
        default=None,
    )

    parser.add_argument(
        "-y",
        help=(
            "Filename of known y data (if available). "
            "If not available then metrics are not returned."
        ),
        type=str,
        default=None,
    )

    parser.add_argument(
        "-v", "--verbose", help="Verbose flag", action="store_true", default=False
    )

    return parser.parse_args()


def read_X(X_filename):
    """Read features from CSV filename.  Features must match model features.

      Arguments:
      X_filename {str} -- [description]

      Returns:
      [dataframe] -- Pandas dataframe of features. Must match model.
    """

    X = pd.read_csv(X_filename, index_col="passengerid")
    return X


def read_y(y_filename):
    """Read known y values from CSV file.

    Arguments:
      y_filename {str} -- Filename of CSV file of known y values.

    Returns:
      y {dataframe} -- Pandas dataframe of known y values.
    """

    if y_filename is not None:
        y = pd.read_csv(y_filename, index_col="passengerid")
    else:
        y = None

    return y


def write_kaggle_submission_output_file(y_pred, filename):
    """ Write a Kaggle submission file. 

        >> head -5 model__logres.csv

            PassengerId,Survived
            892,0
            893,1
            894,0
            895,0
            ...

    Arguments:
      y_pred {dataframe} -- Predicted y labels.
      filename {str} -- CSV filename containing known labels. 
    """

    if filename is not None:
        y_pred.to_csv(filename)


def predict(model, X):
    """ Predict y from model and X

    Arguments:
      model {scikit learn model object} -- [description]
      X {dataframe} -- [description]

    Returns:
      y_pred {dataframe} -- Predicted labels from model.
    """
    y_pred = pd.Series(model.predict(X), index=X.index, name="Survived").to_frame()
    y_pred.index.names = ["PassengerId"]

    return y_pred


def measure_accuracy(y_known, y_predicted):
    """ Measure accuracy of known and predicted labels. If this function is called
        the predicted accuracy score will be displayed to standard out. 

    Arguments:
        y_known {dataframe} -- [description]
        y_predicted {dataframe} -- [description]

    Returns:
        [float] -- Returns the predicted accuracy score. 
    """

    predicted_accuracy_score = metrics.accuracy_score(y_known, y_predicted)
    print(f"\nAccuracy Score on X_test,y_test: {predicted_accuracy_score: 0.4f}\n")

    return predicted_accuracy_score


def run_model():
    """ Main function for command line interface.
    """

    in_args = argparse_command_line()

    model = load(in_args.model)
    X = read_X(in_args.X)
    y_known = read_y(in_args.y)

    # --- Check assertions
    # assert len(model.coef_[0]) == len(X.columns)

    # --- Predict model and write output file for Kaggle submission.
    y_pred = predict(model, X)
    if in_args.output_filename is not None:
        write_kaggle_submission_output_file(y_pred, in_args.output_filename)

    # --- Measure and report accuracy if y_known available.
    if y_known is not None:
        assert (X.index == y_pred.index).all()
        measure_accuracy(y_known, y_pred)


if __name__ == "__main__":
    sys.exit(run_model())
