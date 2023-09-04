import pandas as pd
import numpy as np
import nltk
import re
import os
import string
import pickle
import warnings
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from PIL import Image
from IPython.display import Image, display

RANDOM_STATE = 42

nltk.download("vader_lexicon")
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 500)

# Suppress ConvergenceWarnings from scikit-learn
warnings.filterwarnings("ignore", category=Warning)


# Setup dataframe
# From internet
# url='https://drive.google.com/file/d/1eEtlmdLUTZnyY34g9bL5D3XuWLzSEqBU/view?usp=sharing'
# path ='https://drive.google.com/uc?id=' + url.split('/')[-2]
# df = pd.read_csv(path)

# url="https://drive.google.com/file/d/1x2Tdn1UGhQ6x08yfchUykUJvid257vFY/view?usp=sharing"
# path ='https://drive.google.com/uc?id=' + url.split('/')[-2]
# df2 = pd.read_csv(path)

# # # Locally Home PC
# df = pd.read_csv("D:/Programming/DB's/Positive_negative_reviews_classification/reviews.csv")
# df2 = pd.read_csv("D:/Programming/DB's/Positive_negative_reviews_classification/labels.csv")

# # Locally work PC
df = pd.read_csv(
    "D:/Programming/db's/Positive_negative_reviews_classification_data/reviews.csv"
)
df2 = pd.read_csv(
    "D:/Programming/db's/Positive_negative_reviews_classification_data/labels.csv"
)

# Create a mapping dictionary
mapping = {"Positive": 1, "Negative": 0}

# Use the map function to replace values
df2.sentiment = df2.sentiment.map(mapping)


# ### Preprocessing

# #### Duplicates
# * we drop the duplicate
df.drop(121, axis=0, inplace=True)
df2.drop(121, axis=0, inplace=True)

# ## Analysis


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


# Text cleaned
df["text"] = df["text"].apply(clean_text)

# ## Feature Generation

# Split the data into training and testing sets (80% train, 20% test)
train_X, test_X, train_y, test_y = train_test_split(
    df[["text"]], df2["sentiment"], test_size=0.2, random_state=RANDOM_STATE
)


# ### Additional features generation


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


# ### TF-IDF Feature Generation


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


# ### Saving Metrics to File (.jpg)


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


# ## Training


def train_and_evaluate_model(
    model_type,
    hyperparameters,
    train_X,
    train_y,
    test_X,
    test_y,
    max_features=50,
    features="all",
):
    """
    Train and evaluate a machine learning model.

    Parameters:
    - model_type (str): The type of the model to train ('logisticregression', 'svc', 'xgbclassifier').
    - hyperparameters (dict): Hyperparameters for model tuning.
    - train_X (array-like): Training features.
    - train_y (array-like): Training labels.
    - test_X (array-like): Test features.
    - test_y (array-like): Test labels.
    - max_features (int): Maximum number of features for TF-IDF vectorization (default: 50).
    - features (str): Type of features to use ('all' or 'tfidf').

    Returns:
    - best_model: The best trained model.
    """
    # transform text to numeric features
    train_X = tfidf(train_X, max_features=max_features, features=features)
    test_X = tfidf(test_X, max_features=max_features, features=features)

    # Create a "models" folder if it doesn't exist
    models_folder = "models"
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Start time
    start = datetime.now()

    # Define the model based on the given model type
    if model_type == "logisticregression":
        model = LogisticRegression(random_state=42, max_iter=50)
    elif model_type == "svc":
        model = SVC(probability=True, random_state=42)
    elif model_type == "xgbclassifier":
        model = XGBClassifier(random_state=42)

    # Create a pipeline with feature normalization and the chosen model
    pipeline = make_pipeline(StandardScaler(with_mean=False, with_std=False), model)

    # RandomizedSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=hyperparameters, scoring="roc_auc", cv=3
    )

    grid_search.fit(train_X, train_y)

    best_pipeline = grid_search.best_estimator_
    best_model = best_pipeline.named_steps[model_type]
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # End time
    end = datetime.now()

    # Calculate elapsed time
    elapsed_time = end - start

    # Extract minutes and seconds
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)

    # Specify the model name
    model_name = f"{model_type.capitalize()} Model"

    # Save the best model as a pickle file
    model_filename = os.path.join(models_folder, f"{model_name}_model.pkl")
    with open(model_filename, "wb") as model_file:
        pickle.dump(best_model, model_file)

    # Predict train dataset
    pred_y = best_pipeline.predict(train_X)

    # Calculate classification report, confusion matrix, AUC for train dataset
    report = classification_report(train_y, pred_y)
    conf_matrix = confusion_matrix(train_y, pred_y)
    auc = roc_auc_score(train_y, pred_y)

    # Train metrics to jpg
    metrics(
        text=f"Train set: {model_name} with TF-IDF + Additional features",
        report=report,
        conf_matrix=conf_matrix,
        auc=auc,
        minutes=minutes,
        seconds=seconds,
        filename=f"{model_name}_train_metrics.jpg",
    )

    # Predict test dataset
    pred_y = best_pipeline.predict(test_X)

    # Calculate classification report, confusion matrix, AUC for test dataset
    report = classification_report(test_y, pred_y)
    conf_matrix = confusion_matrix(test_y, pred_y)
    auc = roc_auc_score(test_y, pred_y)

    # Test metrics to jpg
    metrics(
        text=f"Test set: {model_name} with TF-IDF + Additional features",
        report=report,
        conf_matrix=conf_matrix,
        auc=auc,
        minutes=minutes,
        seconds=seconds,
        filename=f"{model_name}_test_metrics.jpg",
    )

    return best_model


def load_model(model_name):
    # Specify the folder where models are saved
    models_folder = "models"

    # Construct the full file path for the model
    model_filename = os.path.join(models_folder, f"{model_name}_model.pkl")

    # Load the model from the file
    with open(model_filename, "rb") as model_file:
        loaded_model = pickle.load(model_file)

    return loaded_model


# Define hyperparameters for the chosen model
logreg_hyperparameters = {
    "logisticregression__C": [0.1, 1, 5],
    # Add hyperparameters for other models as needed
}

# Define hyperparameters for SVC
svc_hyperparameters = {
    "svc__C": [0.1, 1],  # Adjust the C values as needed
    "svc__kernel": ["rbf"],  # , 'linear'],  # Adjust the kernel choices as needed
}

# Define hyperparameters for XGBClassifier
xgb_hyperparameters = {
    "xgbclassifier__learning_rate": [
        0.01,
        0.1,
        0.3,
    ],  # Adjust the learning rates as needed
    "xgbclassifier__max_depth": [3, 4, 5],  # Adjust the max_depth values as needed
    # Add other hyperparameters for XGBClassifier as needed
}

if __name__ == "__main__":
    # Call the function to train and evaluate the LogReg model
    trained_model = train_and_evaluate_model(
        "logisticregression", logreg_hyperparameters, train_X, train_y, test_X, test_y
    )
    print("NEXT ARCHITECTURE")
    # Call the function to train and evaluate the SVC model
    trained_svc_model = train_and_evaluate_model(
        "svc", svc_hyperparameters, train_X, train_y, test_X, test_y
    )
    print("NEXT ARCHITECTURE")
    # Call the function to train and evaluate the XGBClassifier model
    trained_xgb_model = train_and_evaluate_model(
        "xgbclassifier", xgb_hyperparameters, train_X, train_y, test_X, test_y
    )
