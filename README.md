📉 Principal Component Analysis (PCA) Projects with Scikit-learn
This repository contains two machine learning projects focused on PCA (Principal Component Analysis) for dimensionality reduction. Each project uses real-world digit image datasets and demonstrates a different purpose for PCA:

Digits Dataset → PCA for 2D Visualization

MNIST Dataset → PCA for Dimensionality Reduction (retain 95% variance)

🧰 Requirements
Make sure to install the following Python packages:
pip install numpy pandas matplotlib scikit-learn

Note: On some systems, you may also need to install a GUI backend for Matplotlib (TkAgg is used in the first script).

📁 Project Descriptions
1. 🖼️ PCA Visualization on Digits Dataset
File: pca_digits_visualization.py
Goal: Reduce 64-dimensional image data to 2D and visualize digit classes.

Key Steps:
Load Scikit-learn's built-in Digits dataset (8x8 grayscale images of digits 0–9)

Apply PCA(n_components=2) to reduce features to 2D

Scatter plot each digit class using a unique color

Print the explained variance ratio and assess PCA quality

Output:
2D scatter plot of the 10 digit classes

Explained variance summary:
Explained variance: [0.1489 0.1370]
Sum of Explained variance: 0.2859
n_components is not good

2. 🔢 PCA Dimensionality Reduction on MNIST Dataset
File: pca_mnist_reduction.py
Goal: Use PCA to reduce dimensionality of the 784-feature MNIST dataset while retaining 95% of the variance.

Key Steps:
Fetch the full MNIST dataset from OpenML (28x28 grayscale images)

Standardize feature values using StandardScaler

Split into 85% training and 15% test sets

Apply PCA using PCA(0.95) to automatically select the number of components that retain 95% of the variance

Print feature shape before and after PCA

Output:
Shape after Standardization: (59500, 784)
Shape after PCA: (59500, XX)  # XX depends on how many components keep 95% variance (usually around 150–200)


📊 Summary Table
| Project              | Dataset       | PCA Goal                  | Output                            |
| -------------------- | ------------- | ------------------------- | --------------------------------- |
| Digits Visualization | Digits (8x8)  | Reduce to 2D for plotting | Scatter plot + Variance explained |
| MNIST Dim Reduction  | MNIST (28x28) | Keep 95% variance         | Reduced feature size (\~150 dims) |


📌 Notes
Digits project is ideal for understanding how PCA clusters different classes visually in 2D.

MNIST project is more practical, showing how to compress high-dimensional data while preserving most of its structure.

PCA is unsupervised and doesn't use the label (target) information during transformation.

👨‍💻 Author
Educational PCA examples for dimensionality reduction and visualization using Scikit-learn.
