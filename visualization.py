import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, x_col, y_col):
    """Visualize clusters using a scatter plot."""
    if 'Cluster' not in df:
        print("Error: No clustering information found. Run K-Means first.")
        return
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df[x_col], y=df[y_col], hue=df['Cluster'], palette="viridis", s=50)
    plt.title(f"Clusters based on {x_col} and {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="Cluster")
    plt.show()

if __name__ == "__main__":
    from data_processing import load_data, clean_data
    from clustering import apply_kmeans
    
    df = load_data()
    if df is not None:
        df = clean_data(df)
        df, _ = apply_kmeans(df)
        plot_clusters(df, "Age", "Chol")  # Example columns