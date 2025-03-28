import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
from data_processing import load_data, clean_data
from clustering import apply_kmeans
from training import train_model
from correlation import show_correlation
from visualization import plot_clusters

class HeartHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Health Analysis")
        self.root.geometry("500x400")
        
        self.df = None
        self.model = None
        
        # Buttons
        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.correlation_button = tk.Button(root, text="Show Correlation", command=self.show_correlation, state=tk.DISABLED)
        self.correlation_button.pack(pady=10)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        self.cluster_button = tk.Button(root, text="Apply K-Means", command=self.apply_clustering, state=tk.DISABLED)
        self.cluster_button.pack(pady=10)

        self.plot_button = tk.Button(root, text="Visualize Clusters", command=self.plot_clusters, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(pady=10)
    
    def load_data(self):
        """Load CSV file into a DataFrame."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = load_data(file_path)
            self.df = clean_data(self.df)
            messagebox.showinfo("Success", "Data loaded and cleaned successfully!")
            self.correlation_button.config(state=tk.NORMAL)
            self.train_button.config(state=tk.NORMAL)
            self.cluster_button.config(state=tk.NORMAL)

    def show_correlation(self):
        """Show correlation heatmap."""
        if self.df is not None:
            show_correlation(self.df)
        else:
            messagebox.showerror("Error", "Please load the dataset first.")

    def train_model(self):
        """Train the model and display accuracy + predictions in a scrollable pop-up."""
        if self.df is not None:
            self.model, acc, report, conf_matrix, predictions = train_model(self.df)

            # Create a new scrollable window
            results_window = tk.Toplevel(self.root)
            results_window.title("Model Training Results")
            results_window.geometry("600x500")  # Larger window

            # Create a scrollable text widget
            text_area = scrolledtext.ScrolledText(results_window, wrap=tk.WORD, width=70, height=25)
            text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            # Prepare output text
            output_text = f"Model Accuracy: {acc:.2f}\n\n"
            output_text += "Classification Report:\n" + report + "\n\n"
            output_text += "Confusion Matrix:\n" + str(conf_matrix) + "\n\n"
            output_text += "Sample Predictions:\n" + predictions.head().to_string(index=False)

            # Insert text into scrollable area
            text_area.insert(tk.END, output_text)
            text_area.config(state=tk.DISABLED)  # Make text read-only

        else:
            messagebox.showerror("Error", "Please load the dataset first.")
    def apply_clustering(self):
        """Apply K-Means clustering and add cluster labels to the dataset."""
        if self.df is not None:
            self.df, _ = apply_kmeans(self.df)
            messagebox.showinfo("Success", "K-Means clustering applied!")
            self.plot_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Please load the dataset first.")

    def plot_clusters(self):
        """Visualize clusters using scatter plot."""
        if self.df is not None and "Cluster" in self.df.columns:
            plot_clusters(self.df, "Age", "Chol")  # Example features
        else:
            messagebox.showerror("Error", "Please apply K-Means clustering first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartHealthApp(root)
    root.mainloop()