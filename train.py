import pandas as pd
import nltk
import os

import pickle
import warnings
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from utils.utils_inference import tfidf, clean_text
from utils.utils_train import metrics, load_dataset

RANDOM_STATE = 42

nltk.download("vader_lexicon")
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 500)

# Suppress ConvergenceWarnings from scikit-learn
warnings.filterwarnings("ignore", category=Warning)

# Choose the dataset location
dataset_location = "internet"  # Change this to 'internet', 'home', or 'work' as needed
df, df2 = load_dataset(dataset_location)

# Create a mapping dictionary
mapping = {"Positive": 1, "Negative": 0}

# Use the map function to replace values
df2.sentiment = df2.sentiment.map(mapping)

# Drop Duplicates
df.drop(121, axis=0, inplace=True)
df2.drop(121, axis=0, inplace=True)

# Text cleaned
df["text"] = df["text"].apply(clean_text)

# Feature Generation

# Split the data into training and testing sets (80% train, 20% test)
train_X, test_X, train_y, test_y = train_test_split(
    df[["text"]], df2["sentiment"], test_size=0.2, random_state=RANDOM_STATE
)


# Training
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
    model_name = f"{model_type}"

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
