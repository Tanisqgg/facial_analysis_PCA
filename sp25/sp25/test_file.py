from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from facial_analysis import load_and_center_dataset, get_covariance, get_eig, get_eig_prop, \
    project_and_reconstruct_image, display_image, perturb_image

# Add this at the bottom of facial_analysis.py to test, then remove before submission.
if __name__ == "__main__":
    # Load and center the dataset
    X = load_and_center_dataset('celeba_60x50.npy')
    print("Dataset shape:", X.shape)
    print("Mean after centering:", np.mean(X))  # Should be close to 0

    # Compute covariance
    S = get_covariance(X)
    print("Covariance matrix shape:", S.shape)

    # Get top 50 eigenvalues and vectors
    Lambda, U = get_eig(S, 50)
    print("Eigenvalues (top 5):", np.diag(Lambda)[:5])

    # Get eigenvectors explaining > 0.07 proportion of variance
    Lambda_prop, U_prop = get_eig_prop(S, 0.07)
    print("Eigenvalues (prop):", np.diag(Lambda_prop))

    # Choose an image to project and reconstruct
    celeb_idx = 67
    x = X[celeb_idx]
    x_fullres = np.load('celeba_218x178x3.npy')[celeb_idx]
    reconstructed = project_and_reconstruct_image(x, U)

    # Display images
    fig, ax1, ax2, ax3 = display_image(x_fullres, x, reconstructed)
    plt.show()

    # Perturb image
    x_perturbed = perturb_image(x, U, sigma=1000)
    fig, ax1, ax2, ax3 = display_image(x_fullres, x, x_perturbed)
    plt.show()