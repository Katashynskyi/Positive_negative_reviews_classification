import pandas as pd
import numpy as np
import re
import os
import string
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_model(model_name):
    # Specify the folder where models are saved
    models_folder = "models"

    # Construct the full file path for the model
    model_filename = os.path.join(models_folder, f"{model_name}")

    # Load the model from the file
    with open(model_filename, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model


def tfidf(X, max_features=50, features: str = "all"):
    """
    Generate TF-IDF (Term Frequency-Inverse Document Frequency) features and optionally stack them with other features.

    Args:
        X (pd.DataFrame): The input DataFrame containing a 'text' column.
        max_features (int): The maximum number of TF-IDF features to generate.
        features (str): Specifies which features to return.
            - 'all': Returns all features (scaled and TF-IDF).
            - 'tfidf': Returns only TF-IDF features.

    Returns:
        np.ndarray: An array of selected features based on the 'features' parameter.

    This function generates TF-IDF features based on the text content in the 'text' column of the input DataFrame (X).
    The generated TF-IDF features can be stacked with other features (scaled features) or returned separately based on the 'features' parameter.

    Note: The 'text' column in the input DataFrame will be used for TF-IDF feature extraction.
    """
    scaled_features = feature_generator(X)

    # Inin vectorizer
    tf_idf = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=max_features,
    )

    # Create TF-IDF features
    tfidf_features = tf_idf.fit_transform(X.text).toarray()

    # Stack them vertically
    stacked_array = np.hstack((scaled_features, tfidf_features))
    if features == "tfidf":
        out = tfidf_features
    elif features == "all":
        out = stacked_array
    return out


def feature_generator(df):
    """
    Generate and append various text-based features to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'text' column.

    Returns:
        np.ndarray: An array of scaled features (including original and derived features).

    This function calculates and appends the following features to the input DataFrame:
    - Count of punctuations in each text
    - Count of words in uppercase in each text
    - Count of words in title case in each text
    - Sentiment scores (negative, neutral, positive, compound) for each text
    - Word count in each text
    - Count of unique words in each text
    - Count of letters in each text
    - Number of stopwords in each text
    - Mean word length in each text
    - Word count percent (unique words) in each text
    - Punctuation percent in each text

    The function then scales these features and returns them as an array.

    Note: The 'text' column in the input DataFrame will be modified to include cleaned text.
    """
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

    df["text"] = df["text"].apply(clean_text)
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
        lambda x: len([w for w in str(x).lower().split() if w in ENGLISH_STOP_WORDS])
    )

    # Average length of the words
    df.loc[:, "mean_word_len"] = df["text"].apply(
        lambda x: round(np.mean([len(w) for w in str(x).split()]), 2)
    )

    # Derived features

    # Word count percent in each comment:
    df.loc[:, "word_unique_percent"] = (
        df.loc[:, "count_unique_word"] * 100 / df["word_count"]
    )
    # Punct percent in each comment:
    df.loc[:, "punct_percent"] = (
        df.loc[:, "count_punctuations"] * 100 / df["word_count"]
    )

    # Scale features
    scaled_features = StandardScaler().fit_transform(df.drop(columns=["text"]))
    # ↑↑↑ we need to scale them before we stack them with tf-idf features (which we shouldn't scale at all).

    return scaled_features


def clean_text(text):
    text = text.lower()

    # Define the pattern to match "Sent from..." and everything after it
    cleaned_text = re.sub(r"Sent from.*$", "", text)

    # Define the pattern to match "Order" followed by numbers and spaces Use re.sub() to replace the pattern with "order"
    cleaned_text = re.sub(r"Order\s*\(#\d+\)", "order number", text)

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
