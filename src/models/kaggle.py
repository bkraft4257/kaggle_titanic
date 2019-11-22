#!/usr/bin/env python

import pandas as pd
import subprocess

KAGGLE_SUBMISSION_TEMPLATE = "../data/raw/kaggle_gender_submission.csv"


def submit_to_kaggle_titanic_competition(filename, message, verbose=True):
    """Submits a CSV file with a message to Kaggle's Titanic Competition.
    The file Before the file is uploaded

    Arguments:
        filename {str} -- CSV filename that you want to submit to Kaggle.

        message {str} -- The message that you want to include in the
                         Kaggle submission.

    Keyword Arguments:
        verbose {bool} --  (default: {True})

    Returns:
        stdout - Output string from the subprocess module
        stderr - Output string from the subprocess module
    """
    stdout, stderr = None, None

    try:
        if is_valid_kaggle_submission(filename, message):
            stdout, stderr = upload_kaggle_titanic_submission_via_api(filename, message)
    except:
        print("Kaggle submission of {filename} failed.")

        print(stdout)
        print(stderr)

        raise

    return stdout, stderr


def is_valid_kaggle_submission(filename, message):
    """
    Verify that the file you are submitting to Kaggle is valid for that
    competition.
    """

    y_pred_file = pd.read_csv(filename).set_index("PassengerId")

    # The code below is to test that you have a valid submission
    y_submission = pd.read_csv(KAGGLE_SUBMISSION_TEMPLATE).set_index("PassengerId")

    is_index_correct = (y_pred_file.index == y_submission.index).all()
    is_index_names_correct = y_pred_file.index.names == y_submission.index.names
    is_column_names_correct = (y_pred_file.columns == y_submission.columns).all()

    assert is_index_correct
    assert is_index_names_correct
    assert is_column_names_correct

    return is_index_correct & is_index_names_correct & is_column_names_correct


def upload_kaggle_titanic_submission_via_api(filename, message):
    """Upload CSV to Kaggle Titanic Competition via Kaggle API.

        Arguments:
        filename {str} -- CSV filename that you want to submit to Kaggle.

        message {str} -- The message that you want to include in the
                         Kaggle submission.

    Returns:
        stdout {str} -- Standard Output during subprocess submission.
        stderr {str} -- Standard Error during subprocess submission.
    """

    process = subprocess.Popen(
        ["kaggle", "competitions", "submit", "titanic", "-f", filename, "-m", message],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    print(filename)
    print(message)
    print(stdout)
    print(stderr)

    return stdout, stderr
