from utils.utils_inference import load_model, tfidf, clean_text
import argparse
import pandas as pd


def main(input_csv_file, output_csv_file, model_file="logisticregression_model.pkl"):
    df = pd.read_csv(input_csv_file)
    df["text"] = df["text"].apply(clean_text)
    test_X = tfidf(df[["text"]], max_features=50, features="all")
    loaded_model = load_model(f"{model_file}_model.pkl")

    # Make predictions on the test data
    predictions = loaded_model.predict(test_X)

    # Create a mapping dictionary for labels
    label_mapping = {0: "Negative", 1: "Positive"}

    # Replace numeric labels with text labels
    predictions_text = [label_mapping[label] for label in predictions]

    # Create a DataFrame with 'id' and 'sentiment' columns
    output_df = pd.DataFrame(
        {
            "id": df["id"],  # Assuming a column named 'id' in your input data
            "sentiment": predictions_text,  # Use the text labels
        }
    )

    # Save the predictions to the output CSV file
    output_df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict labels using a trained model."
    )
    parser.add_argument(
        "input_csv_file",
        nargs="?",
        default="test_reviews.csv",
        help="Input CSV file containing text data (default: test_reviews.csv).",
    )
    parser.add_argument(
        "output_csv_file",
        nargs="?",
        default="test_labels_pred.csv",
        help="Output CSV file to save predictions (default: predictions.csv).",
    )
    parser.add_argument(
        "model_file",
        nargs="?",
        default="xgbclassifier",
        help="Trained model file. Choose from 'logisticregression', 'svc', or 'xgbclassifier'. (default: 'xgbclassifier')",
    )

    args = parser.parse_args()

    main(args.input_csv_file, args.output_csv_file, args.model_file)
    print("test_labels_pred.csv file generated, congrats!")
