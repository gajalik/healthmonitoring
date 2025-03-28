import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_for_correlation(df):
    """Convert categorical columns to numeric before correlation analysis."""
    df = df.copy()  # Avoid modifying the original dataset

    # Convert categorical columns using label encoding (simpler)
    categorical_columns = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Ca", "Thal"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes  # Convert categories to numbers

    return df

def show_correlation(df):
    """Display a heatmap of feature correlations after converting categorical columns."""
    df_numeric = preprocess_for_correlation(df)  # Convert categorical data

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    from data_processing import load_data, clean_data

    df = load_data()
    if df is not None:
        df = clean_data(df)
        show_correlation(df)