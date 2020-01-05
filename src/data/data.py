import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from nameparser import HumanName
from sklearn.preprocessing import scale
from typing import Optional

from IPython.display import display
from collections import Counter

import great_expectations as ge


class ExtractData:
    """Extract Titanic data from the Kaggle's train.csv file.
    """

    def __init__(self, filename: Union[str, Path], age_bins=None, drop_columns=None):
        """Extract Training Data from filename (string or Path object)

        Arguments:
            filename {[str]} -- Filename of CSV data file containing data.
            drop_columns -- Columns in dataframe that should be dropped.
        """
        if drop_columns is None:
            drop_columns = ["age", "cabin", "name", "ticket"]

        self.filename = filename

        self.drop_columns = drop_columns
        self.all_label_columns = ["survived"]
        self.all_feature_columns = [
            "pclass",
            "name",
            "sex",
            "age",
            "sibsp",
            "parch",
            "ticket",
            "fare",
            "cabin",
            "embarked",
        ]
        self.Xy_raw = None
        self.extract_raw()

    def extract_raw(self):
        """
        Extracts data from a CSV file.

        Returns:
            pd.DataFrame -- [description]
        """
        Xy_raw = pd.read_csv(self.filename)

        Xy_raw.columns = Xy_raw.columns.str.lower().str.replace(" ", "_")

        Xy_raw = Xy_raw.rename(columns={"age": "age_raw"})
        Xy_raw = Xy_raw.rename(columns={"fare": "fare_raw"})

        Xy_raw["pclass"] = Xy_raw["pclass"].astype("category")
        self.Xy_raw = Xy_raw.set_index("passengerid")


class TransformData:

    """
    TransformData takes the raw extracted data cleans and creates new features before
    returning a new dataframe.

    The training and test data contain the following:
        * 1 Lady. She was traveling with a sibling and no husband. Set title to Miss
        * 2 Mlle and 1 Mme. All 3 were 24 years old and traveling alone.  Retitled as Miss.
        * 1 Sir. Male 49 years old. traveling with a sibling.
        * Revs were all males.
        * 8 Drs. (7 male, 1 female) changed to Mr. and Mrs. respectively.
    """

    translate_title_dictionary = {
        "Mlle.": "Miss.",
        "Mme.": "Miss.",
        "Sir.": "Mr.",
        "Ms.": "Mrs.",
        "Rev.": np.nan,
        "Col.": "Mr.",
        "Capt.": "Mr.",
        "Lady.": "Miss.",
        "the Countess. of": "Mrs.",
        "Dr.": np.nan,
    }

    def __init__(
        self,
        raw_data,
        adult_age_threshold_min=13,
        age_bins=None,
        age_bin_label=None,
        embarked_mode=None,
        xy_age_estimate=None,
        xy_fare_estimate=None,
        drop_columns=None,
        translate_title_dictionary=None,
        fare_bins=None,
        fare_bin_labels=None,
    ):
        """Transform Data according to the rules established in the EDA. To apply
        the same rules to another data set you must explicitly pass in

        * adult_age_threshold_min
        * age_bins
        * embarked_mode
        * xy_age_estimate

        Arguments:
             filename [str|Path] -- Filename of CSV data file containing data.
             drop_columns -- Columns in dataframe that should be dropped.
        """
        if translate_title_dictionary is None:
            translate_title_dictionary = self.translate_title_dictionary

        if age_bins is None:
            # Old age_bins are here.
            # age_bins = [0, 10, 20, 30, 40, 50, 60, np.inf]
            age_bins = (0, 5, 12, 18, 25, 35, 60, 120)

        if age_bin_label is None:
            age_bin_label = [
                "baby",
                "child",
                "teen",
                "student",
                "young_adult",
                "adult",
                "senior",
            ]

        if fare_bin_labels is None:
            fare_bin_labels = ["q1", "q2", "q3", "q4"]

        assert len(age_bins) == len(age_bin_label) + 1

        self.translate_title_dictionary = translate_title_dictionary

        self.raw = raw_data
        self.adult_age_threshold_min = adult_age_threshold_min
        self.drop_columns = drop_columns
        self.age_bins = age_bins
        self.age_bin_label = age_bin_label

        self.Xy = self.raw.Xy_raw.copy()

        self.Xy_age_estimate = xy_age_estimate

        self.Xy_fare_estimate = xy_fare_estimate

        self.fare_bins = fare_bins
        self.fare_bin_labels = fare_bin_labels

        if embarked_mode is None:
            embarked_mode = self.Xy["embarked"].mode()[0]

        self.embarked_mode = embarked_mode

    # Methods

    def transform(self):
        self.Xy["embarked"] = impute_missing_embarked(
            self.Xy.embarked, self.embarked_mode
        )
        self.Xy["title"] = extract_title(
            self.Xy[["name", "sex"]], self.translate_title_dictionary
        )
        self.Xy["last_name"] = extract_last_name(self.Xy.name)
        self.Xy["family_size"] = calc_family_size(self.Xy[["sibsp", "parch"]])

        self.reset_fare_equals_0_nan()

        if self.Xy_fare_estimate is None:
            self.Xy_fare_estimate = estimate_by_group(
                self.Xy, "fare_raw", ["pclass"], "median", 2
            ).rename("fare_estimate")

        self.impute_missing_fare()
        self.calc_fare_bins()

        if self.Xy_age_estimate is None:
            self.Xy_age_estimate = estimate_by_group(
                self.Xy, "age_raw", ["sex", "title"], "mean", 1
            ).rename("age_estimate")

        self.impute_missing_age()
        self.calc_age_bins()

        self.Xy["is_child"] = calc_is_child(
            self.Xy["age"], self.adult_age_threshold_min
        )

        self.Xy["is_traveling_alone"] = calc_is_traveling_alone(self.Xy["family_size"])

        self.Xy = self.Xy.sort_index()

    def calc_age_bins(self):
        """Calculates age bins.
        """
        self.Xy["age_bin"] = pd.cut(
            self.Xy.age, bins=self.age_bins, labels=self.age_bin_label
        )

    def calc_fare_bins(self):
        """Calculates fare bins.

           If fare_bins is None then calculate the fare_bins based upon the
           quartiles. If fare_bins is a list then calculate the fair_bin using
           pd.cut()
        """

        if self.fare_bins is None:
            self.Xy["fare_bin"] = pd.qcut(
                self.Xy.fare.fillna(-1),
                q=[0, 0.25, 0.5, 0.75, 1.0],
                labels=self.fare_bin_labels,
            )

            self.fare_bins = (
                [0] + self.Xy.groupby(["fare_bin"]).fare.max().tolist()[0:-1] + [1000]
            )

        else:
            self.Xy["fare_bin"] = pd.cut(
                self.Xy.fare, bins=self.fare_bins, labels=self.fare_bin_labels
            )

        assert (len(self.fare_bins) - 1) == len(self.fare_bin_labels)

    def reset_fare_equals_0_nan(self):
        """

        """
        self.Xy["fare_raw"] = self.Xy["fare_raw"].replace(0, np.nan)

    def estimate_fare(self, groupby_columns="pclass"):
        """Estimate fare for passengers that travelled for free (fare = 0).
        This is based upon the assumption that no passengers traveled for free
        and that a fare=0 means that information was missing.  Estimate the NaN
        fares with median of the fare.

        Keyword Arguments:
            groupby_columns {str} -- The columns that will be grouped over to
            estimate the fare.
        """
        if self.Xy_fare_estimate is None:
            self.Xy_fare_estimate = (
                self.Xy.groupby(groupby_columns).fare_raw.median().to_frame().round(2)
            )

            self.Xy_fare_estimate = self.Xy_fare_estimate.rename(
                columns={"fare_raw": "fare_estimate"}
            )

    def estimate_age(self, groupby_columns=["sex", "title"]):
        """Estimate age of passenger when age is unknown.   The age will be
        estimated according to the group as specified in the groupby_columns.

        Keyword Arguments:
            groupby_columns {list} -- [description] (default: {["sex", "title"]})
        """

        if self.Xy_age_estimate is None:
            self.Xy_age_estimate = (
                self.Xy.groupby(groupby_columns).age_raw.mean().to_frame().round(1)
            )

            self.Xy_age_estimate = self.Xy_age_estimate.rename(
                columns={"age_raw": "age_estimate"}
            )

    def impute_missing_age(self):

        groupby_columns = list(self.Xy_age_estimate.index.names)

        out_df = (
            self.Xy.reset_index()
            .merge(
                self.Xy_age_estimate.reset_index(),
                on=groupby_columns,
                how="left",
                indicator=False,
            )
            .set_index("passengerid")
        )

        out_df["age"] = out_df["age_raw"].fillna(out_df["age_estimate"])

        self.Xy = out_df

    def impute_missing_fare(self):
        """Imputes missing fare based upon only the most frequent fare. This
        could be improved by looking at additional features. In particular,
        the number of passengers in the party and pclass.
        """
        groupby_columns = list(self.Xy_fare_estimate.index.names)

        out_df = (
            self.Xy.reset_index()
            .merge(
                self.Xy_fare_estimate, on=groupby_columns, how="left", indicator=False
            )
            .set_index("passengerid")
        )

        out_df["fare"] = out_df["fare_raw"].fillna(out_df["fare_estimate"])

        self.Xy = out_df

    # =============================================================================


TRANSLATE_TITLE_DICTIONARY = {
    "Mlle.": "Miss.",
    "Mme.": "Miss.",
    "Sir.": "Mr.",
    "Ms.": "Mrs.",
    "Rev.": np.nan,
    "Col.": "Mr.",
    "Capt.": "Mr.",
    "Lady.": "Miss.",
    "the Countess. of": "Mrs.",
    "Dr.": np.nan,
}


def extract_title(in_df: pd.DataFrame, translate_title_dictionary=None) -> pd.Series:
    """
    Extract title from the name using nameparser.
    If the Title is empty then we will fill the title with either Mr or Mrs depending upon the sex.  This
    is adequate for the train and holdout data sets.  The title being empty only occurs for passenger 1306
    in the holdout data set.

    Args:
        in_df:
        translate_title_dictionary:

    Returns:

    """

    assert "name" in in_df.columns
    assert "sex" in in_df.columns

    if translate_title_dictionary is None:
        translate_title_dictionary = TRANSLATE_TITLE_DICTIONARY

    out_df = in_df.copy()

    title = (
        out_df.name.apply(lambda x: HumanName(x).title)
        .replace({r"\.": ""}, regex=True)
        .replace(translate_title_dictionary)
        .replace({"": np.nan})
        .fillna(out_df["sex"])
        .replace({"female": "Mrs", "male": "Mr"})
    )

    out_df["title"] = title

    return out_df.title


def calc_family_size(in_df: pd.DataFrame) -> pd.Series:
    """
    Create feature family size, which is the number of people (including
    self) that are traveling together.

    Arguments:
        in_df: Dataframe containing columns sibsp and parch

    Returns:
        Dataframe with the new column family_size, where family size
        is sibsp + parch + 1
    """

    return pd.Series(in_df.sibsp + in_df.parch + 1, name="family_size")


def extract_last_name(in_series: pd.Series) -> pd.Series:
    """
    Extracts last name from name featureusing nameparser from a series.

    Arguments:
        in_df {pd.Series} -- [description]

    Returns:
        Pandas series with only the last name.
    """
    assert in_series.name == "name"

    out_series = in_series.apply(lambda x: HumanName(x).last)
    out_series.name = "last_name"

    assert out_series.name == "last_name"

    return out_series


def calc_is_child(in_series: pd.Series, adult_age_threshold_min: int = 13) -> pd.Series:
    """
    Calculate Boolean feature if passenger is a child as determined by the self.adult_age_threshold_min

    Args:
        in_series:
        adult_age_threshold_min:

    Returns:

    """

    assert in_series.name == "age"
    assert adult_age_threshold_min >= 0

    return pd.Series(in_series < adult_age_threshold_min)


def impute_missing_embarked(
    in_series: pd.Series, assume_embarked_from: Optional[str] = None
) -> pd.Series:
    """
    Imputes missing embarkment location based upon the most frequent
    place to embark.

    Args:
        in_series:
        assume_embarked_from: C=Cherbourg, Q=Queenstown, S=Southampton

    Returns:

    """

    if assume_embarked_from is None:
        assume_embarked_from = in_series.mode()[0]

    assert assume_embarked_from in ["C", "Q", "S"]

    return in_series.fillna(assume_embarked_from)


def calc_is_traveling_alone(in_series: pd.Series):

    """Create Boolean feature if passenger is traveling alone. (True=Traveling alone, False=Traveling in group)
    """
    assert in_series.name == "family_size"

    return in_series == 1


def estimate_by_group(
    in_df: pd.DataFrame,
    column_to_estimate,
    groupby_columns,
    method="mean",
    round_to_decimal=2,
) -> pd.Series:
    """
    Estimate fare for passengers that travelled for free (fare = 0).
    This is based upon the assumption that no passengers traveled for free
    and that a fare=0 means that information was missing.  Estimate the NaN
    fares with median of the fare.

    Args:
        in_df:
        column_to_estimate: The column to estimate.
        groupby_columns:  The columns that will be grouped to estimate the desired column by group.

    Returns: pd.Series

    """
    """

    Keyword Arguments:
        groupby_columns {str} --
    """
    if method == "mean":
        out_df = (
            in_df.groupby(groupby_columns)[column_to_estimate]
            .mean()
            .round(round_to_decimal)
        )
    else:
        out_df = (
            in_df.groupby(groupby_columns)[column_to_estimate]
            .median()
            .round(round_to_decimal)
        )

    return out_df


def impute_missing_metric(
    in_series,
    in_estimate,
    column_name_of_metric_with_na,
    column_name_of_estimated_metric,
    column_name_of_final_metric,
):

    groupby_columns = list(in_estimate.index.names)
    index_names = list(in_series.index.names)

    out_df = (
        in_series.reset_index().merge(
            in_estimate.reset_index(), on=groupby_columns, how="left", indicator=False
        )
    ).set_index(index_names)

    out_df[column_name_of_final_metric] = out_df[column_name_of_metric_with_na].fillna(
        out_df[column_name_of_estimated_metric]
    )

    return out_df[column_name_of_final_metric]


def transform_X_numerical(Xy, columns=["age", "fare", "family_size"]):

    # Scale the numerical columns.
    return pd.DataFrame(scale(Xy[columns]), index=Xy.index, columns=columns)


def transform_X_categorical(
    Xy,
    columns=[
        "sex",
        "embarked",
        "title",
        "age_bin",
        "fare_bin",
        "is_child",
        "is_traveling_alone",
    ],
    drop_first=True,
):

    # Encode the categorical features. The first category will be dropped.
    return pd.get_dummies(Xy[columns], drop_first=drop_first)


def transform_X(
    Xy,
    numerical_columns=["age", "fare", "family_size"],
    categorical_columns=[
        "sex",
        "embarked",
        "title",
        "age_bin",
        "fare_bin",
        "is_child",
        "is_traveling_alone",
        "pclass",
    ],
):

    # Scale the numerical columns.
    X_numerical = transform_X_numerical(Xy, numerical_columns)

    # Encode the categorical features. The first category will be dropped.
    X_cat_encoded = transform_X_categorical(Xy, categorical_columns)

    return X_numerical.join(X_cat_encoded)
