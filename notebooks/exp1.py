# Import necessary libraries for data handling, machine learning, tracking, and visualization
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner='sivakumar026', repo_name='mlops_pipeline', mlflow=True)

mlflow.set_experiment("Experiment 1")

mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/mlops_pipeline.mlflow"
)

# Load the dataset from a CSV file
data = pd.read_csv("../data/water_potability.csv")

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

# Function to fill missing values with median
def fill_missing_with_median(df):

    for column in df.columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column].fillna(median_value, inplace=True)

    return df

# Fill missing values
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Import RandomForestClassifier and pickle
from sklearn.ensemble import RandomForestClassifier
import pickle

# Features and target
X_train = train_processed_data.drop(columns=["Potability"], axis=1)

y_train = train_processed_data["Potability"]

n_estimators = 100

# Start MLflow run
with mlflow.start_run():

    # Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators
    )

    clf.fit(X_train, y_train)

    # Save model
    pickle.dump(clf, open("model.pkl", "wb"))

    # Test data
    X_test = test_processed_data.iloc[:, 0:-1].values

    y_test = test_processed_data.iloc[:, -1].values

    # Metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )

    # Load model
    model = pickle.load(open('model.pkl', "rb"))

    # Prediction
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("acc", acc)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1-score", f1)

    # Log parameter
    mlflow.log_param("n_estimators", n_estimators)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 5))

    sns.heatmap(cm, annot=True)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    # Save confusion matrix
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix artifact
    mlflow.log_artifact("confusion_matrix.png")

    # Classification report
    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:

        f.write(report)

    # Log classification report artifact
    mlflow.log_artifact("classification_report.txt")

    # Log model.pkl artifact
    mlflow.log_artifact("model.pkl")

    # Log sklearn model
    mlflow.sklearn.log_model(
        clf,
        "RandomForestClassifier"
    )

    # Log source code
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tag("author", "datathinkers")

    mlflow.set_tag("model", "RF")

    # Print metrics
    print("Accuracy:", acc)

    print("Precision:", precision)

    print("Recall:", recall)

    print("F1-score:", f1)