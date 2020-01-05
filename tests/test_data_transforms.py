import pandas as pd
import numpy as np
import pytest
from src.data.data import (
    extract_last_name,
    calc_family_size,
    extract_title,
    calc_is_child,
    impute_missing_embarked,
)

test_data_1 = [
    (
        ["Robert Kraft", "Elizabeth Granger", "Betsy Smith", "Madeline Brown"],
        ["Kraft", "Granger", "Smith", "Brown"],
    ),
    (
        [
            "Robert Arthur Kraft",
            "Elizabeth Ann Granger",
            "Betsy Natalie Smith",
            "Madeline Margaret Brown",
        ],
        ["Kraft", "Granger", "Smith", "Brown"],
    ),
    (
        ["R. Kraft", "E. Granger", "B. Smith", "M. Brown"],
        ["Kraft", "Granger", "Smith", "Brown"],
    ),
]


@pytest.mark.parametrize(
    "in_data, expected",
    test_data_1,
    ids=["first_last", "first_middle_last", "first_initial_last_name"],
)
def test_extract_last_name_series(in_data, expected):
    in_series = pd.Series(data=in_data, name="name")
    out_series = extract_last_name(in_series)
    expected_series = pd.Series(expected, name="lastname")

    assert in_series.name == "name"
    assert out_series.name == "last_name"
    assert out_series.equals(expected_series)


def test_calc_family_size():
    in_df = pd.DataFrame(data={"sibsp": [1, 2, 3], "parch": [0, 0, 1]})
    out_df = calc_family_size(in_df)
    expected_df = pd.Series(data=[2, 3, 5], name="family_size")

    assert expected_df.equals(out_df)


def test_extract_title():
    in_df = pd.DataFrame(data={"name": ["Dr. Robert A. Kraft"], "sex": ["male"]})
    out_df = extract_title(in_df)
    expected_df = pd.DataFrame(
        data={"name": ["Dr. Robert A. Kraft"], "sex": ["male"], "title": ["Dr"]}
    )

    assert expected_df.equals(out_df)


test_data__calc_is_child = [
    ([12, 13, 14, 15, 16], [True, False, False, False, False], 13),
    ([12, 13, 14, 15, 16], [True, True, True, False, False], 15),
    ([12, 13, 14, 15, 16], [True, True, True, True, False], 15.5),
]


@pytest.mark.parametrize(
    "in_data, expected, threshold",
    test_data__calc_is_child,
    ids=["age_threshold_13", "age_threshold_15", "age_threshold_15.5_float"],
)
def test_calc_is_child(in_data, expected, threshold):
    in_series = pd.Series(in_data, name="age")
    out_series = calc_is_child(in_series, threshold)
    expected_series = pd.Series(expected, name="is_child")

    assert expected_series.equals(out_series)


def test_calc_is_child_negative_threshold():
    in_series = pd.Series([12, 13, 14, 15, 16], name="age")

    with pytest.raises(AssertionError):
        calc_is_child(in_series, -1)


def test_impute_missing_embarked_with_autofill():
    in_series = pd.Series(["C", "S", "S", "Q", np.nan], name="embarked")
    out_series = impute_missing_embarked(in_series)
    expected_series = pd.Series(["C", "S", "S", "Q", "S"], name="embarked")

    assert expected_series.equals(out_series)


def test_impute_missing_embarked():
    in_series = pd.Series(["C", "S", "S", "Q", np.nan], name="embarked")
    out_series = impute_missing_embarked(in_series, "Q")
    expected_series = pd.Series(["C", "S", "S", "Q", "Q"], name="embarked")

    assert expected_series.equals(out_series)


def test_impute_missing_embarked_with_none_missing():
    in_series = pd.Series(["C", "S", "S", "Q", "Q"], name="embarked")
    out_series = impute_missing_embarked(in_series)
    expected_series = pd.Series(["C", "S", "S", "Q", "Q"], name="embarked")

    assert expected_series.equals(out_series)


def test_impute_missing_embarked_with_incorrect_embarked():
    in_series = pd.Series(["C", "S", "S", "Q", np.nan], name="embarked")

    with pytest.raises(AssertionError):
        impute_missing_embarked(in_series, "X")
