import numpy as np
from sklearn.cluster import KMeans

def apply_kmeans(df, n_clusters=3):
    """Apply K-Means clustering to the dataset."""
    # Selecting numerical features for clustering
    X = df.select_dtypes(include=[np.number])  
    
    # Applying K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    return df, kmeans

if __name__ == "__main__":
    from data_processing import load_data, clean_data
    
    df = load_data()
    if df is not None:
        df = clean_data(df)
        df, model = apply_kmeans(df)
        print(df.head())