# Facial Analysis using PCA

This project implements Principal Component Analysis (PCA) to perform facial image analysis, including image reconstruction and perturbation experiments. It's ideal for applications in facial recognition, compression, and exploratory data analysis.

## Description

The repository includes Python scripts for loading, preprocessing, and analyzing facial image data. Key functionalities provided:

- **Dataset Preprocessing:** Loading and centering facial image datasets.
- **Covariance Matrix Computation:** Creating covariance matrices necessary for PCA.
- **Eigendecomposition:** Computing eigenvalues and eigenvectors from covariance matrices.
- **Variance-Based Selection:** Selecting eigenvectors based on explained variance.
- **Image Projection and Reconstruction:** Reducing dimensionality and reconstructing images from PCA components.
- **Image Perturbation:** Testing the robustness of PCA reconstruction by adding Gaussian noise.

## Files Included

- `facial_analysis.py`: Core functions for dataset processing, PCA computations, image reconstruction, and perturbation.
- `test_file.py`: Example script demonstrating the usage of functionalities provided in `facial_analysis.py`.
- `.npy` data files (`celeba_60x50.npy` and `celeba_218x178x3.npy`): Facial image datasets for testing.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies using:

```bash
pip install numpy scipy matplotlib
```

## Usage

To use the provided scripts, execute the following from the project root:

```bash
python test_file.py
```

This script demonstrates:

- Dataset loading and centering
- Covariance matrix calculation
- PCA eigendecomposition
- Image projection, reconstruction, and perturbation

## Example Outputs

Running `test_file.py` will:

1. Display original high-resolution images.
2. Show PCA-reconstructed images.
3. Illustrate the effect of perturbations with Gaussian noise.

## Contributing

Contributions are welcome! Feel free to submit issues, suggest improvements, or send pull requests.

## License

This project is licensed under the MIT License.
