import numpy as np
import pandas as pd # Included for general data science context, though not strictly used for DataFrames here
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt # Standard way to import pyplot

# Note: matplotlib.use('TkAgg') is often used for specific interactive backends.
# For a GitHub-friendly script, it's generally best practice to remove it
# unless there's a specific requirement for that backend, as Matplotlib
# can usually select an appropriate one for the environment.

def load_digits_data():
    """
    Loads the handwritten digits dataset from scikit-learn.

    This dataset consists of 1797 8x8 pixel grayscale images of handwritten digits (0-9).
    Each image is a numerical representation of an integer (0-9).

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data (features),
                             target (labels), images, and other metadata.
    """
    print("Loading handwritten digits dataset...")
    digits = datasets.load_digits()
    print(f"Dataset loaded: {digits.data.shape[0]} samples, {digits.data.shape[1]} features (8x8 pixels flattened).")
    print("Target classes (digits):", digits.target_names)
    return digits

def perform_pca(data, n_components=2):
    """
    Performs Principal Component Analysis (PCA) on the given data.

    PCA is a dimensionality reduction technique that transforms a dataset
    into a new set of dimensions (principal components) that are orthogonal
    and capture the maximum variance in the data.

    Args:
        data (numpy.ndarray): The input feature data.
        n_components (int): The number of principal components to retain.

    Returns:
        tuple: A tuple containing:
            - x_pca (numpy.ndarray): The data projected onto the principal components.
            - estimator (sklearn.decomposition.PCA): The fitted PCA estimator object.
    """
    print(f"\nPerforming PCA with n_components={n_components}...")
    estimator = PCA(n_components=n_components)
    x_pca = estimator.fit_transform(data)
    print(f"Original data shape: {data.shape}")
    print(f"PCA-transformed data shape: {x_pca.shape}")

    # Explained variance shows how much information (variance) can be attributed
    # to each principal component.
    explained_variance_ratio = estimator.explained_variance_ratio_
    print("Explained variance ratio for each component:", explained_variance_ratio)
    sum_explained_variance = np.sum(explained_variance_ratio)
    print(f'Sum of Explained Variance (total variance captured): {sum_explained_variance:.4f}')

    # Check if the retained components capture a significant amount of variance
    if sum_explained_variance < 0.95:
        print('Warning: The selected n_components captures less than 95% of the total variance. '
              'Consider increasing n_components if you need more information.')
    else:
        print('The selected n_components captures a good amount (>= 95%) of the total variance.')

    return x_pca, estimator

def visualize_pca_results(x_pca, y_data, target_names):
    """
    Visualizes the PCA-transformed data in a 2D scatter plot.
    Each digit class is represented by a different color.

    Args:
        x_pca (numpy.ndarray): The data projected onto 2 principal components.
        y_data (numpy.ndarray): The original target labels.
        target_names (list): List of names corresponding to target labels (digits 0-9).
    """
    print("\n--- Visualizing PCA Results ---")
    plt.figure(figsize=(10, 8))
    # Define colors for each digit (0-9) for visualization
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    # Ensure we have enough colors for all unique target classes
    unique_targets = np.unique(y_data)
    if len(colors) < len(unique_targets):
        print(f"Warning: Not enough colors defined for all {len(unique_targets)} target classes. "
              "Some classes might share colors or be uncolored.")
        # Extend colors if necessary, or cycle through them
        colors_extended = colors * (len(unique_targets) // len(colors) + 1)
    else:
        colors_extended = colors

    for i in unique_targets:
        # Select data points belonging to the current digit class 'i'
        # x_pca[:, 0] is the first principal component, x_pca[:, 1] is the second
        px = x_pca[:, 0][y_data == i]
        py = x_pca[:, 1][y_data == i]
        plt.scatter(px, py, c=colors_extended[i], label=f'Digit {target_names[i]}', alpha=0.7)

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D PCA of Handwritten Digits Dataset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == "__main__":
    # 1. Load the Digits Dataset
    digits_dataset = load_digits_data()
    X = digits_dataset.data
    y = digits_dataset.target

    # 2. Perform PCA to reduce dimensions to 2 components for visualization
    # This transforms the 64-dimensional data into a 2-dimensional space.
    X_pca_transformed, pca_estimator = perform_pca(X, n_components=2)

    # 3. Visualize the PCA results
    visualize_pca_results(X_pca_transformed, y, digits_dataset.target_names)

    print("\nScript execution complete.")
