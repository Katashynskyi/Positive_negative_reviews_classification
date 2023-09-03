import emoji
import numpy as np
import pandas as pd
import re
import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Hardcoded meta-class of TFIDF features
    Preprocessor to transform raw text data into a sparse matrix of features.

    Returns:
    --------
    df : csr_matrix
        TF-IDF embedding combined with additional features
    """

    def __init__(self, tfidf_on=True, tf_idf_max_features=100):
        self.tfidf_on = tfidf_on
        self.tf_idf_max_features = tf_idf_max_features
        self.df = None

    def _feature_gen_before_clean_text(self, df):
        # punctuation count
        df.loc[:, "count_punctuations"] = df["text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation])
        )  # including "I'll" as punctuation
        # upper case words count
        df.loc[:, "count_words_upper"] = df["text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )
        # title case words count
        df.loc[:, "count_words_title"] = df["text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()])
        )
        return df

    def _clean_text(self, text):
        # # make all comments str type
        # text = str(text)

        # Convert emoji to their textual representations
        text = emoji.demojize(text)

        # Convert text to lowercase
        text = text.lower()

        # Replace URLs with a placeholder
        text = re.sub(r"http\S+", "<URL>", text)

        # Define the pattern to match "Sent from..." and everything after it
        text = re.sub(r"Sent from.*$", "", text)

        # Define the pattern to match "Order" followed by numbers and spaces Use re.sub() to replace the pattern with "order"
        text = re.sub(r"Order\s*\(#\d+\)", "order number", text)

        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub("\W", " ", text)  # != a-z, A-Z, and 0-9
        text = re.sub(
            "\s+", " ", text
        )  # s(spaces)== \t \n \r (return carret on the beginning)

        text = re.sub("<.*?>", " ", text)
        text = text.translate(str.maketrans(" ", " ", string.punctuation))

        # Keep ! and ? symbols
        text = re.sub("[^a-zA-Z!?]", " ", text)
        text = re.sub("\n", " ", text)
        text = text.strip(" ")

        return text

    def _feature_gen_after_clean_text(self, df):
        # punctuation count
        sentiment = SentimentIntensityAnalyzer()

        # Initialize empty lists for each sentiment score
        neg_scores = []
        neu_scores = []
        pos_scores = []
        compound_scores = []

        # Loop through the sentences and calculate sentiment scores
        for sentence in df["text"]:
            ss = sentiment.polarity_scores(sentence)
            neg_scores.append(ss["neg"])
            neu_scores.append(ss["neu"])
            pos_scores.append(ss["pos"])
            compound_scores.append(ss["compound"])

        # Add sentiment scores as new columns to the DataFrame
        df["negative"] = neg_scores
        df["neutral"] = neu_scores
        df["positive"] = pos_scores
        df["compound"] = compound_scores

        # Word count in each comment:
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))

        # Unique word count
        df.loc[:, "count_unique_word"] = df["text"].apply(
            lambda x: len(set(str(x).split()))
        )

        # Letter count
        df.loc[:, "count_letters"] = df["text"].apply(lambda x: len(str(x)))

        # Number of stopwords
        df.loc[:, "count_stopwords"] = df["text"].apply(
            lambda x: len(
                [w for w in str(x).lower().split() if w in ENGLISH_STOP_WORDS]
            )
        )

        # Average length of the words
        df.loc[:, "mean_word_len"] = df["text"].apply(
            lambda x: round(np.mean([len(w) for w in str(x).split()]), 2)
        )

        ### Derived features

        # Word count percent in each comment:
        df.loc[:, "word_unique_percent"] = (
            df.loc[:, "count_unique_word"] * 100 / df["word_count"]
        )
        # Punct percent in each comment:
        df.loc[:, "punct_percent"] = (
            df.loc[:, "count_punctuations"] * 100 / df["word_count"]
        )
        self.df = df.iloc[:, 1]
        df.drop(columns=["text", "id"], inplace=True)
        return df

    def _scaler(self, df) -> pd.DataFrame:
        "Normalize 14 features"
        self.scaled_features = StandardScaler().fit_transform(df)
        return self.scaled_features

    def _tfidf(self, tfidf_and_add_features=True, tf_idf_max_features=100):
        # Inin vectorizer
        tf_idf = TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=tf_idf_max_features,
        )

        # Create TF-IDF features
        tfidf_features_only = tf_idf.fit_transform(self.df).toarray()

        # Stack them vertically
        stacked_array = np.hstack((self.scaled_features, tfidf_features_only))

        if tfidf_and_add_features == True:
            out = stacked_array
        elif tfidf_and_add_features == False:
            out = tfidf_features_only
        return out

    def fit(self, X, y=None):
        # Dummy
        return self

    def transform(self, df, tfidf_on=True, tf_idf_max_features=100):
        # Step 1
        df = self._feature_gen_before_clean_text(df)  # (254, 5)

        # Step 2
        df["text"] = df["text"].apply(lambda text: self._clean_text(text))  # (254, 5)
        # Step 3
        df = self._feature_gen_after_clean_text(df)  # (254, 14)
        df = self._scaler(df)  # numpy.ndarray (254, 14)

        # Step 4
        if tfidf_on == True:
            df = self._tfidf(
                tfidf_and_add_features=True, tf_idf_max_features=tf_idf_max_features
            )
        return df

    def fit_transform(self, X, y=None):
        tfidf_on = self.tfidf_on
        tf_idf_max_features = self.tf_idf_max_features
        return self.transform(
            X, tfidf_on=tfidf_on, tf_idf_max_features=tf_idf_max_features
        )


if __name__ == "__main__":
    # Import train dataset
    url = "https://drive.google.com/file/d/1eEtlmdLUTZnyY34g9bL5D3XuWLzSEqBU/view?usp=sharing"
    path1 = "https://drive.google.com/uc?id=" + url.split("/")[-2]
    df = pd.read_csv(path1)

    df = Preprocessor(tfidf_on=True, tf_idf_max_features=100).fit_transform(df)
    print(df)
    print(df.shape)
