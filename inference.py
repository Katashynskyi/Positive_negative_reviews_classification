import sys
import pandas as pd
from train import load_model

# Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python inference.py input_csv_file output_csv_file")
    sys.exit(1)  # Exit with an error code

# Extract the input and output file names from command-line arguments
input_csv_file = sys.argv[1]
output_csv_file = sys.argv[2]

# Load the input CSV file (test_reviews.csv in this case)
try:
    df = pd.read_csv(input_csv_file)
except FileNotFoundError:
    print(f"Error: File '{input_csv_file}' not found.")
    sys.exit(1)

print(df)

# Load a model (replace 'LogisticRegression' with the desired model name)
loaded_logreg_model = load_model("Logisticregressio Model")
type(loaded_logreg_model)
# Perform your inference or prediction logic on the loaded data (df) here
# This is where you would typically process the test reviews and generate predictions

# Save the predictions or results to the output CSV file (test_labels_pred.csv in this case)
# For demonstration, let's create a simple DataFrame with dummy predictions
dummy_predictions = pd.DataFrame(
    {
        "id": df["id"],  # Assuming a column named 'id' in your input data
        "PredictedLabel": [0]
        * len(df),  # Dummy predictions, replace with your actual predictions
    }
)

dummy_predictions.to_csv(output_csv_file, index=False)
