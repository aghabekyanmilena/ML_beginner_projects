# Machine Learning and Data Processing Exercises

This Jupyter Notebook contains various foundational exercises and mini-projects in machine learning, image processing, and numerical analysis.
It includes both custom implementations and applications of standard libraries like `scikit-learn`, `PyTorch`.

## Contents

### 1. Polynomial Fitting

* **Functionality**: Fits an n-th order polynomial to given data points using NumPy's `polyfit` and evaluates it with `polyval`.
* **Goal**: Understand and apply polynomial regression to a small dataset.

### 2. Logistic Regression from Scratch

* **Functionality**: Implements logistic regression using gradient descent.
* **Datasets Used**:

  * Synthetic dataset generated via `make_classification`
  * Custom-generated n-dimensional sphere classification dataset
* **Loss Function**: Cross-entropy
* **Performance Metric**: Accuracy

### 3. Logistic Regression using scikit-learn

* **Dataset**: Iris dataset
* **Library**: `scikit-learn`
* **Purpose**: Comparison between a custom implementation and a standard library implementation of logistic regression.

### 4. N-Dimensional Sphere Classification

* **Goal**: Classify whether a point in n-dimensional space lies inside the unit sphere.
* **Techniques**:

  * Synthetic data generation using Euclidean norm
  * Logistic regression model from scratch

### 5. Image Quantization and PCA Visualization

* **Image**: Parrot (`parrot.jpeg`)
* **Tasks**:

  * Reduce the number of colors using K-Means clustering
  * Reduce image dimensions using PCA
  * Visualize results with `matplotlib`

### Prerequisites

Make sure you have the following libraries installed:

```bash
pip install numpy scikit-learn torch torchvision matplotlib pillow
```

### Running the Notebook

1. Open `tasks.ipynb` using Jupyter or Google Colab.
2. Run all cells sequentially.
3. Make sure to have `parrot.jpeg` in the same directory or upload it in the Colab environment.

This notebook is meant for educational purposes and demonstrates the implementation of ML algorithms from scratch.

This notebook was created as part of a machine learning assignment, showcasing proficiency in custom implementations and standard ML workflows.
