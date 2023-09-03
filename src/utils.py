import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Preset settings
RANDOM_STATE = 42


# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)


class Split:
    """
    Splitting pd.DataFrame into training & testing sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be split into training and testing sets.
    test_size : float, optional (default=0.1)
        The proportion of the DataFrame to use for testing.

    Returns:
    --------
    .get_train_data(): -> tuple[pd.DataFrame, pd.DataFrame
        X_train, y_train : A separated DataFrame
    or

    .get_test_data(): -> tuple[pd.DataFrame, pd.DataFrame]
        X_test, y_test : A separated DataFrame
    """

    def __init__(self, df_path=None, target_path=None, test_size: float = 0.2):
        self._df = df_path
        self._target = target_path

        self._test_size = test_size

        self._X_train = self._y_train = pd.DataFrame(), pd.DataFrame()
        self._X_test = self._y_test = pd.DataFrame(), pd.DataFrame()

    def _split(self):
        """
        Processing the input DataFrame into train and test sets.
        Use get_train_data & get_test_data methods .

        Returns:
            self
        """
        # Read
        _df = pd.read_csv(self._df)
        _target = pd.read_csv(self._target)

        # Create a mapping dictionary
        mapping = {"Positive": 1, "Negative": 0}

        # Use the map function to replace values
        _target = _target["sentiment"].map(mapping)

        # Shuffle
        _df = shuffle(_df, random_state=RANDOM_STATE)
        _target = shuffle(_target, random_state=RANDOM_STATE)

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            _df["text"],
            _target,
            stratify=_target,
            test_size=self._test_size,
            random_state=RANDOM_STATE,
        )
        self._X_train = self._X_train.to_frame()
        self._X_test = self._X_test.to_frame()
        self._y_train = self._y_train.to_frame()
        self._y_test = self._y_test.to_frame()

        return self


    def get_train_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_train & y_train
        """
        self._split()
        # return self._X_train, self._y_train
        return self._X_train, self._y_train

    def get_test_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_test & y_test
        """
        self._split()
        return self._X_test, self._y_test


if __name__ == "__main__":
    # Path
    path = "D:/Programming/DB's/Positive_negative_reviews_classification/reviews.csv"
    target = "D:/Programming/DB's/Positive_negative_reviews_classification/labels.csv"

    # Split test
    splitter = Split(df_path=path, target_path=target)
    train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
    # print(train_X)
    test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
    print(f"train_X:\n{train_X.tail(1)}")
    print(f"train_y:\n{train_y.tail(1)}")
    print(f"test_X:\n{test_X.tail(1)}")
    print(f"test_y:\n{test_y.tail(1)}")
