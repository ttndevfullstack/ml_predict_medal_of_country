import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def clean_data(data, required_columns=None):
    print("👉 Cleaning data...")
    if required_columns is None:
        required_columns = [
            "team",
            "country",
            "year",
            "athletes",
            "age",
            "prev_medals",
            "medals",
        ]

    # Check if required columns exist in the DataFrame
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(
            f"The following columns are missing from the DataFrame: {missing_columns}"
        )

    cleaned_data = data[required_columns].dropna()
    print("✅ Cleaning completed successfully...")

    return cleaned_data


def load_data(file_path="teams.csv"):
    print(f"👉 Starting to load data from path: {file_path}...")
    try:
        teams = pd.read_csv(file_path)
        print("✅ Loading data completed successfully...")

        return teams
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: There was a problem parsing the file '{file_path}'.")
        return None


def extract_training_data(data):
    return data[data["year"] < 2012]


def extract_testing_data(data):
    return data[data["year"] >= 2012]

def prepare_data(data):
    print("👉 Prepare data to train and test the model...")
    training_data = extract_training_data(data)
    testing_data = extract_testing_data(data)
    print("✅ Prepare data completed successfully.....")

    return training_data, testing_data

def predict_medals(trained_model, testing_data):
    print("👉 Starting predicting the medals...")
    predictions = trained_model.predict(testing_data)

    # Replace negative predictions with 0
    predictions = np.maximum(predictions, 0)
    
    # Round predictions to the integer
    predictions = np.round(predictions).astype(int)
    print("✅ Predicting completed successfully...")

    return predictions


def training_linear_model(training_data, predictors, target):
    print("👉 Training the linear regression model...")
    trained_model = LinearRegression().fit(training_data[predictors], training_data[target])
    print("✅ Training completed successfully...")
    return trained_model

def calculateResult(testing_data):
    num_mean_abs_error = mean_absolute_error(testing_data["medals"], testing_data["predictions"])
    abs_errors = (testing_data["medals"] - testing_data["predictions"]).abs()
    abs_error_by_teams = abs_errors.groupby(testing_data["team"]).mean()
    medals_by_team = testing_data["medals"].groupby(testing_data["team"]).mean()
    error_ratio = abs_error_by_teams / medals_by_team
    error_ratio = error_ratio[np.isfinite(error_ratio)].sort_values()
    mape = error_ratio.mean() * 100
    prediction_accuracy = 100 - mape

    return num_mean_abs_error, mape, prediction_accuracy


def main():
    data_file_path = "teams.csv"
    columns = ["team", "country", "year", "athletes", "age", "prev_medals", "medals"]

    # Load and clean data
    origin_data = load_data(data_file_path)
    cleaned_data = clean_data(origin_data, columns)
    training_data, testing_data = prepare_data(cleaned_data)

    target = "medals"
    predictors = ["athletes", "prev_medals"]

    # Train the model and make predictions
    trained_model = training_linear_model(training_data, predictors, target)
    predictions = predict_medals(trained_model, testing_data[predictors])
    
    # Add predictions to testing data
    new_testing_data = testing_data.copy()
    new_testing_data.loc[:, "predictions"] = predictions

    # Calculate absolute error and percentage error
    num_mean_abs_error, mape, prediction_accuracy = calculateResult(new_testing_data)

    # Print results
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               📊 Model Results                           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║ 🎯 Mean Absolute Error (MAE): {num_mean_abs_error:<20}")
    print(f"║ 🎯 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"║ 🏆 Prediction Accuracy: {prediction_accuracy:.2f}%")
    print("╚══════════════════════════════════════════════════════════╝")

    # print("✅")
    # print("🎯")
    # print("🏆")

    return 0


main()
