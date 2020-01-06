from src.data.data import ExtractData, TransformData, TransformBin

import pandas as pd
import numpy as np
import pytest
from src.data.data import (
    extract_last_name,
    calc_family_size,
    extract_title,
    calc_is_child,
    impute_missing_embarked,
    estimate_by_group,
    impute_missing_metric_by_group,
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
    expected_series = pd.Series(expected, name="last_name")

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
    expected_df = pd.Series(data=["Dr"], name="title")

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


def test_estimate_by_group_age_sex():

    in_df = pd.DataFrame(
        data={
            "age": [10, 10, 20, 20, 20],
            "sex": [0, 1, 0, 1, 0],
            "metric": [10, 20, 30, 40, 50],
        }
    )

    out_series = estimate_by_group(in_df, "metric", ["age", "sex"])

    expected_series = (
        pd.DataFrame(
            data={
                "metric": [10, 20, 40, 40],
                "age": [10, 10, 20, 20],
                "sex": [0, 1, 0, 1],
            }
        )
        .set_index(["age", "sex"])
        .squeeze()
    )

    assert expected_series.equals(out_series)


def test_estimate_by_group_age():
    in_df = pd.DataFrame(
        data={
            "age": [10, 10, 20, 20, 20, 30, 30],
            "sex": [0, 1, 0, 1, 0, 1, 0],
            "metric": [10, 20, 30, 40, 50, 50, 50],
        }
    )

    out_series = estimate_by_group(in_df, "metric", ["age"])

    expected_series = (
        pd.DataFrame(data={"metric": [15, 40, 50], "age": [10, 20, 30]})
        .set_index(["age"])
        .squeeze()
    )

    assert expected_series.equals(out_series)


def test_impute_missing_metric():

    in_df = pd.DataFrame(
        data={
            "age": [10, 10, 20, 20, 20, 30, 30],
            "metric_raw": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, np.nan],
        }
    ).set_index("age")

    metric_estimate = pd.DataFrame(
        data={"metric_estimate": [15.0, 40.0, 50.0], "age": [10, 20, 30]}
    ).set_index("age")

    expected_df = (
        pd.DataFrame(
            data={
                "age": [10, 10, 20, 20, 20, 30, 30],
                "metric_raw": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, np.nan],
                "metric_estimate": [15.0, 15.0, 40.0, 40.0, 40.0, 50.0, 50.0],
                "metric": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0],
            }
        )
        .set_index("age")
        .squeeze()
    )

    out_df = impute_missing_metric_by_group(
        in_df, metric_estimate, "metric_raw", "metric_estimate", "metric"
    )

    assert expected_df.equals(out_df)


def test_transform_bin_metric_raw():

    vector = list(range(0, 101, 10))

    in_df = pd.DataFrame(data={"age": vector, "metric_raw": vector}).set_index("age")

    expected_df = pd.DataFrame(
        data={
            "age": vector,
            "metric_raw": vector,
            "metric_raw_bin": ["q1"] * 3 + ["q2"] * 3 + ["q3"] * 2 + ["q4"] * 3,
        }
    ).set_index("age")

    expected_category = pd.CategoricalDtype(
        categories=["q1", "q2", "q3", "q4"], ordered=True
    )
    expected_df.metric_raw_bin = expected_df.metric_raw_bin.astype(expected_category)

    tb = TransformBin(in_df, "metric_raw")

    out_df = tb.transform()

    assert (tb.bins == [0, 25, 50, 75, 100]).all()
    assert tb.bin_qcut == [0, 0.25, 0.5, 0.75, 1.0]
    assert tb.bin_labels == ["q1", "q2", "q3", "q4"]

    assert type(out_df) == pd.DataFrame
    assert type(expected_df) == pd.DataFrame
    assert expected_df.equals(out_df)


def test_transform_bin_age():

    vector = list(range(0, 101, 10))

    in_df = pd.DataFrame(data={"age": vector, "metric_raw": vector})

    expected_df = pd.DataFrame(
        data={
            "age": vector,
            "metric_raw": vector,
            "age_bin": ["q1"] * 3 + ["q2"] * 3 + ["q3"] * 2 + ["q4"] * 3,
        }
    )

    expected_category = pd.CategoricalDtype(
        categories=["q1", "q2", "q3", "q4"], ordered=True
    )
    expected_df.age_bin = expected_df.age_bin.astype(expected_category)

    tb = TransformBin(in_df, "age")

    out_df = tb.transform()

    assert (tb.bins == [0, 25, 50, 75, 100]).all()
    assert tb.bin_qcut == [0, 0.25, 0.5, 0.75, 1.0]
    assert tb.bin_labels == ["q1", "q2", "q3", "q4"]

    assert type(out_df) == pd.DataFrame
    assert type(expected_df) == pd.DataFrame
    assert expected_df.equals(out_df)


# =========================================
# Tests for Refactoring TransformData class

translate_title_dictionary = {
    "Mlle": "Miss",
    "Mme": "Miss",
    "Sir": "Mr",
    "Ms": "Mrs",
    "Rev": np.nan,
    "Col": "Mr",
    "Capt": "Mr",
    "Lady": "Miss",
    "the Countess of": "Mrs",
    "Dr": np.nan,
}

age_bins = (0, 5, 12, 18, 25, 35, 60, 120)
age_bin_label = ["baby", "child", "teen", "student", "young_adult", "adult", "senior"]

assert len(age_bins) == len(age_bin_label) + 1


def read_csv_expected_data():
    filename = "../tests/data/expected_transformed_kaggle_train.csv"

    test_train_transformed = pd.read_csv(filename, index_col="passengerid")

    test_train_transformed.pclass = test_train_transformed.pclass.astype("category")

    fare_bin_cat_type = pd.CategoricalDtype(
        categories=["q1", "q2", "q3", "q4"], ordered=True
    )
    test_train_transformed.fare_bin = test_train_transformed.fare_bin.astype(
        fare_bin_cat_type
    )

    age_bin_cat_type = pd.CategoricalDtype(
        categories=[
            "baby",
            "child",
            "teen",
            "student",
            "young_adult",
            "adult",
            "senior",
        ],
        ordered=True,
    )
    test_train_transformed.age_bin = test_train_transformed.age_bin.astype(
        age_bin_cat_type
    )

    return test_train_transformed


def test_refactor_transformed_data():
    train = ExtractData("../data/raw/kaggle_train.csv")
    train.Xy_raw["fare_raw"] = train.Xy_raw["fare_raw"].replace(0, np.nan)

    transformed_train = TransformData(
        train,
        translate_title_dictionary=translate_title_dictionary,
        age_bins=age_bins,
        xy_age_estimate=None,
        age_bin_label=age_bin_label,
    )

    transformed_train.transform()
    expected_train = read_csv_expected_data()

    assert expected_train.equals(transformed_train.Xy)
