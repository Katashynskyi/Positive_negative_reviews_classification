# Positive and Negative Reviews Classification
<img src="https://goldmandarin.com.ua/image/catalog/1_PRODUKT/FEM/favo/wordcloud.jpg" alt="Pic" width="665" height="665">

This project aims to classify text-based reviews as either positive or negative using machine learning models. It includes exploratory data analysis, data preprocessing, feature engineering, model training, and evaluation.

### Prerequisites

- Python 3.9+
- Virtual environment (Conda 3.9)
- Required libraries (specified in `requirements.txt`)

### Overview

This project includes an Exploratory Data Analysis (EDA) Jupyter/Colab notebook located at EDA/EDA.ipynb. The EDA notebook generates a wordcloud.jpg and creates the following folders for models and metrics:

    models: Contains trained machine learning models.
    metric_pics: Contains metrics visualizations.

The project's root directory contains the following main components:

    train.py: Generates models and metrics visuals, saving them to their respective folders.

    inference.py: By default, this script uses the XGBoost Classifier model for predictions. It generates predictions and saves them in a file named test_labels_pred.csv with id and sentiment columns.

The utils folder includes two Python files:

    utils_train.py: Contains various functions used in train.py.
    utils_inference.py: Contains functions used in inference.py, ensuring no additional library uploads are required.

All additional files placed near the inference script and will be loaded automatically.

### How to Run

1. Drag and Drop test_reviews.csv to the root folder

2. Run the following commands from your virtual environment:

```bash
pip install -r requirements.txt
```
3. Execute the inference script with the following command: (Linux and macOS)
```bash
python3 inference.py test_reviews.csv test_labels_pred.csv
```
or (for Windows)
```bash
python inference.py test_reviews.csv test_labels_pred.csv
```
4. Take your generated test_labels_pred.csv file from the root folder.