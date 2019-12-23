import pandas as pd
import pytest
from src.data.data import extract_last_name

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
