import pandas as pd

def load_data(filepath="heart.csv"):
    """Load heart health dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def clean_data(df):
    """Handle missing values by dropping them."""
    df = df.dropna()
    return df

def get_summary(df):
    """Return basic summary statistics of the dataset."""
    return df.describe()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = clean_data(df)
        print(get_summary(df))