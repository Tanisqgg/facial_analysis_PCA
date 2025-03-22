Facial Analysis using PCA
This project implements a facial analysis pipeline using Principal Component Analysis (PCA). It provides functions to load and center datasets, compute covariance matrices, perform eigen decomposition, reconstruct images, and visualize the results. The code also includes an option to perturb images by adding Gaussian noise.

Features
Dataset Loading and Centering: Load facial datasets stored in .npy format and center the data by subtracting the mean.

Covariance Matrix Computation: Calculate the covariance matrix of the centered dataset.

Eigen Decomposition: Compute eigenvalues and eigenvectors for the covariance matrix. You can either extract a fixed number of components or select those that explain a desired proportion of the variance.

Image Projection and Reconstruction: Project images onto the principal component space and reconstruct them.

Visualization: Display the original high-resolution, original, and reconstructed images using Matplotlib.

Image Perturbation: Optionally perturb the reconstruction by adding Gaussian noise.

Requirements
Python 3.6 or higher

NumPy

SciPy

Matplotlib

You can install the required libraries using pip:

bash
Copy
pip install numpy scipy matplotlib
Getting Started
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Prepare your dataset:

Ensure you have a dataset in .npy format containing facial image data. The script expects the data to be formatted appropriately for the provided functions.

Using the Module:

Import the functions from facial_analysis.py in your Python script or Jupyter Notebook. For example:

python
Copy
import facial_analysis as fa

# Load and center the dataset
data_centered = fa.load_and_center_dataset('your_dataset.npy')

# Compute the covariance matrix
cov_matrix = fa.get_covariance(data_centered)

# Extract the top 10 eigenvalues and eigenvectors
Lambda, U = fa.get_eig(cov_matrix, k=10)

# For a given image, project it and reconstruct using the top eigenvectors
reconstructed_image = fa.project_and_reconstruct_image(image, U)
Visualization
To visualize the original and reconstructed images, use the provided function:

python
Copy
fig, ax1, ax2, ax3 = fa.display_image(high_res_image, original_image, reconstructed_image)
fig.show()
Make sure your image data is reshaped correctly as the function expects specific dimensions.

Contributing
Contributions are welcome! Please open an issue or submit a pull request with any enhancements, bug fixes, or suggestions.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
