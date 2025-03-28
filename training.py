import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_for_training(df):
    """Convert categorical variables into numeric format before training."""
    df = df.copy()  # Avoid modifying the original dataset

    categorical_columns = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Ca", "Thal"]
    label_encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])  # Convert categories to numbers
            label_encoders[col] = le

    return df, label_encoders  # Return label encoders in case we need to inverse-transform later

def train_model(df):
    """Train a logistic regression model after encoding categorical features."""
    df, _ = preprocess_for_training(df)  # Convert categorical variables

    X = df.drop(columns=["Target"])  # Features
    y = df["Target"]  # Target variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Show sample predictions
    predictions_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    print("\nSample Predictions:\n", predictions_df.head())

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", confusion)

    return model, accuracy, report, confusion, predictions_df

if __name__ == "__main__":
    from data_processing import load_data, clean_data

    df = load_data()
    if df is not None:
        df = clean_data(df)
        model, acc, report, conf_matrix, predictions = train_model(df)
        print(f"\nModel Accuracy: {acc:.2f}")