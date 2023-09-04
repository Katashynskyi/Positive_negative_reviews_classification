import pandas as pd
import numpy as np
import nltk
import os
import string

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

from PIL import Image
from IPython.display import Image, display
from utils.utils_inference import clean_text

RANDOM_STATE = 42

nltk.download("vader_lexicon")
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 500)

# Suppress ConvergenceWarnings from scikit-learn
warnings.filterwarnings("ignore", category=Warning)


def load_dataset(dataset_location="internet"):
    """
    Load the dataset from the specified location (internet, home, or work).

    Args:
        dataset_location (str): The location from which to load the dataset.
            - 'internet': Load data from an internet source (Google Drive).
            - 'home': Load data from your home PC.
            - 'work': Load data from your work PC.

    Returns:
        tuple: A tuple containing two pandas DataFrames - (df, df2).
               - df: The dataframe containing text data.
               - df2: The dataframe containing sentiment labels.

    This function loads the dataset based on the specified location and returns two dataframes:
    - df: Contains the text data.
    - df2: Contains the corresponding sentiment labels.

    If an invalid dataset_location is provided, a ValueError is raised."""
    if dataset_location == "internet":
        # Load data from an internet source (Google Drive)
        url = "https://drive.google.com/file/d/1eEtlmdLUTZnyY34g9bL5D3XuWLzSEqBU/view?usp=sharing"
        path = "https://drive.google.com/uc?id=" + url.split("/")[-2]
        df = pd.read_csv(path)

        url2 = "https://drive.google.com/file/d/1x2Tdn1UGhQ6x08yfchUykUJvid257vFY/view?usp=sharing"
        path2 = "https://drive.google.com/uc?id=" + url2.split("/")[-2]
        df2 = pd.read_csv(path2)
    elif dataset_location == "home":
        # Load data from your home PC
        df = pd.read_csv(
            "D:/Programming/DB's/Positive_negative_reviews_classification/reviews.csv"
        )
        df2 = pd.read_csv(
            "D:/Programming/DB's/Positive_negative_reviews_classification/labels.csv"
        )
    elif dataset_location == "work":
        # Load data from your work PC
        df = pd.read_csv(
            "D:/Programming/db's/Positive_negative_reviews_classification_data/reviews.csv"
        )
        df2 = pd.read_csv(
            "D:/Programming/db's/Positive_negative_reviews_classification_data/labels.csv"
        )
    else:
        raise ValueError(
            "Invalid dataset location. Please choose 'internet', 'home', or 'work'."
        )

    return df, df2


# Additional features generation
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


# Saving Metrics to File (.jpg)
def metrics(text, report, conf_matrix, auc, minutes, seconds, filename=None):
    """
    Generate classification metrics visualization and optionally save it as an image.

    Args:
        text (str): A text description or title for the metrics.
        report (str): The classification report.
        conf_matrix (numpy.ndarray): The confusion matrix.
        auc (float): The area under the ROC curve (AUC) score.
        minutes (int): The elapsed time in minutes.
        seconds (float): The elapsed time in seconds (fractions).
        filename (str, optional): The filename to save the visualization as an image. If not provided, the image will be displayed but not saved.

    Returns:
        None or IPython.display.Image: If a filename is provided, None is returned. Otherwise, the metrics visualization is displayed as an Image in the Jupyter Notebook.

    This function generates a classification metrics visualization including a classification report, confusion matrix, AUC score, and elapsed time.
    The visualization can be optionally saved as an image with the specified filename.
    """
    output_text = (
        f"{text}\n\n"
        "Classification Report:\n\n" + report + "\n\n"
        "Confusion Matrix:\n" + str(conf_matrix) + "\n\n"
        "AUC: "
        + str(int(auc * 100) / 100.0)
        + "\n\nElapsed time:\n"
        + f"{minutes} minutes\n{round(seconds, 3)} seconds"
    )

    # Create a fixed-size plot with the output text
    plt.figure(figsize=(8, 6))  # Specify the desired width and height in inches
    plt.text(
        0.9, 0.5, output_text, fontsize=12, ha="right", va="center", wrap=True
    )  # Adjust x-coordinate (0.9) for right alignment
    plt.axis("off")

    if filename:
        # Create a "metrics_pics" folder if it doesn't exist
        metrics_folder = "metrics_pics"
        if not os.path.exists(metrics_folder):
            os.makedirs(metrics_folder)

        # Save the plot as an image in the "metrics_pics" folder with the specified filename
        output_image_path = os.path.join(metrics_folder, filename)
        plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Display the image in the Jupyter Notebook
        return display(Image(filename=output_image_path, width=400))

    # If no filename is specified, display the image without saving it
    return display(plt.gcf())


def preprocess(df, df2):
    """
    Preprocess the input dataframes and split them into training and testing sets.

    Args:
        df (pd.DataFrame): The input dataframe containing text data.
        df2 (pd.DataFrame): The input dataframe containing sentiment labels.

    Returns:
        tuple: A tuple containing four arrays - (train_X, test_X, train_y, test_y).
               - train_X: Training features (text data).
               - test_X: Testing features (text data).
               - train_y: Training labels (sentiment labels).
               - test_y: Testing labels (sentiment labels).

    This function performs the following preprocessing steps:
    - Creates a mapping dictionary to convert sentiment labels to binary values (1 for Positive, 0 for Negative).
    - Uses the map function to replace sentiment label values in df2.
    - Removes duplicates from both input dataframes.
    - Cleans the text data in df by applying the 'clean_text' function.
    - Splits the data into training and testing sets with an 80-20 split ratio using random_state for reproducibility.

    Note: The 'clean_text' function is assumed to be available for text cleaning.
    """
    # Create a mapping dictionary
    mapping = {"Positive": 1, "Negative": 0}

    # Use the map function to replace values
    df2.sentiment = df2.sentiment.map(mapping)

    # Drop Duplicates
    df.drop(121, axis=0, inplace=True)
    df2.drop(121, axis=0, inplace=True)

    # Text cleaned
    df["text"] = df["text"].apply(clean_text)

    # Split the data into training and testing sets (80% train, 20% test)
    train_X, test_X, train_y, test_y = train_test_split(
        df[["text"]], df2["sentiment"], test_size=0.2, random_state=RANDOM_STATE
    )
    return train_X, test_X, train_y, test_y
