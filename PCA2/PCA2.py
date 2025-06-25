import pandas as pd # fetch_openml can return pandas DataFrames/Series
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_mnist_data():
    """
    Fetches the MNIST handwritten digits dataset from OpenML.

    The MNIST database of handwritten digits has a training set of 60,000 examples,
    and a test set of 10,000 examples. It is a subset of a larger set of
    handwritten digits collected from American Census Bureau employees and
    American high school students.

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data (features),
                             target (labels), and other metadata.
    """
    print("Fetching MNIST dataset (mnist_784) from OpenML...")
    # fetch_openml might download the data if not cached.
    # set as_frame=False to get numpy arrays, which is typical for image data preprocessing.
    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    print(f"MNIST dataset loaded: {mnist_data.data.shape[0]} samples, {mnist_data.data.shape[1]} features.")
    print("Target classes (digits):", np.unique(mnist_data.target))
    return mnist_data

def split_data(features, target, train_size=0.85, random_state=0):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The feature data.
        target (numpy.ndarray): The target data.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({train_size*100:.0f}%) and testing ({(1-train_size)*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is important for classification tasks.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, train_size=train_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(features_train)} samples")
    print(f"Testing set size: {len(features_test)} samples")
    return features_train, features_test, target_train, target_test

def standardize_data(features_train, features_test):
    """
    Standardizes the training and testing feature data.

    Standardization (Z-score normalization) transforms data to have a mean of 0 and a
    standard deviation of 1. This is important for many machine learning algorithms,
    especially distance-based ones or those sensitive to feature scales (like PCA).

    Args:
        features_train (numpy.ndarray): Training features.
        features_test (numpy.ndarray): Testing features.

    Returns:
        tuple: A tuple containing (scaled_features_train, scaled_features_test).
    """
    print("\n--- Standardizing Data ---")
    scaler = StandardScaler()
    # Fit the scaler on training data ONLY to prevent data leakage from the test set
    features_train_scaled = scaler.fit_transform(features_train)
    # Apply the same transformation (learned from training data) to the test data
    features_test_scaled = scaler.transform(features_test)

    print("Shape after Standardization (Training):", features_train_scaled.shape)
    print("Shape after Standardization (Testing):", features_test_scaled.shape)
    return features_train_scaled, features_test_scaled

def perform_pca_variance_retention(features_train, features_test, variance_to_retain=0.95):
    """
    Performs Principal Component Analysis (PCA), retaining a specified percentage of variance.

    Instead of fixing the number of components, PCA can be configured to find the minimum
    number of components required to explain a certain amount of variance in the data.

    Args:
        features_train (numpy.ndarray): Standardized training features.
        features_test (numpy.ndarray): Standardized testing features.
        variance_to_retain (float): The desired percentage of variance to retain (e.g., 0.95 for 95%).

    Returns:
        tuple: A tuple containing (pca_features_train, pca_features_test, pca_decomposer).
    """
    print(f"\n--- Reducing Dimensionality with PCA (retaining {variance_to_retain*100:.0f}% variance) ---")
    # Initialize PCA with variance_to_retain. PCA will automatically determine n_components.
    pca_decomposer = PCA(n_components=variance_to_retain)

    # Fit PCA on training data ONLY to prevent data leakage
    features_train_pca = pca_decomposer.fit_transform(features_train)
    # Apply the same PCA transformation (learned from training data) to the test data
    features_test_pca = pca_decomposer.transform(features_test)

    print(f"Number of components selected by PCA to retain {variance_to_retain*100:.0f}% variance: {pca_decomposer.n_components_}")
    print("Shape after PCA (Training):", features_train_pca.shape)
    print("Shape after PCA (Testing):", features_test_pca.shape)
    print(f"Actual explained variance retained: {np.sum(pca_decomposer.explained_variance_ratio_):.4f}")

    return features_train_pca, features_test_pca, pca_decomposer

if __name__ == "__main__":
    # Define parameters for data loading and processing
    TRAIN_TEST_SPLIT_RATIO = 0.85
    RANDOM_SEED = 0 # For reproducibility
    PCA_VARIANCE_RETENTION = 0.95 # Retain 95% of variance

    # 1. Load the MNIST Dataset
    mnist_dataset = load_mnist_data()
    X = mnist_dataset.data
    y = mnist_dataset.target

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        train_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 3. Standardize the Data
    # Essential for PCA to ensure features contribute equally, regardless of their scale.
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # 4. Reduce Dimensionality with PCA (retaining 95% of variance)
    X_train_pca, X_test_pca, pca_model = perform_pca_variance_retention(
        X_train_scaled, X_test_scaled,
        variance_to_retain=PCA_VARIANCE_RETENTION
    )

    print("\n--- Final Processed Data Shapes ---")
    print("Training features shape:", X_train_pca.shape)
    print("Testing features shape:", X_test_pca.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    print("\nScript execution complete. The processed data (X_train_pca, X_test_pca, y_train, y_test) is ready for model training.")
