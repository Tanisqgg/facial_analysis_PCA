from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    """
    loads an .npy dataset and centers it
    :param filename: path to the .npy file
    :return: Centered dataset with retained shape as original
    """
    # load file
    x = np.load(filename)

    # compute the mean and center the dataset
    x_centered = x - np.mean(x, axis=0)
    return x_centered


def get_covariance(dataset):
    """
    Computes the covariance matrix of the dataset.
    :param dataset: Centered dataset.
    :return: Covariance matrix (d x d)
    """

    return np.dot(np.transpose(dataset), dataset) / (dataset.shape[0] - 1)


def get_eig(S, k):
    """
    Perform eigendecomposition on the covariance matrix and return the eigenvalues and eigenvectors in descending order.
    :param S: Covariance matrix
    :param k: Number of eigenvectors and eigenvalues
    :return: Diagonal matrix of eigenvalues, matrix of eigenvectors
    """

    eigenvals, eigenvectors = eigh(S, subset_by_index=[S.shape[0] - k, S.shape[0] - 1])

    # reverse the order of eigenvalues
    eigenvals = eigenvals[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # convert to a diagonal matrix
    Lambda = np.diag(eigenvals)

    return Lambda, eigenvectors


def get_eig_prop(S, prop):
    """
    Compute eigenvalues and eigenvectors that explain more than a given proportion of variance.
    :param S: Covariance matrix
    :param prop:  Cumulative Proportion of variance threshold
    :return: Diagonal matrix of selected eigenvalues, and corresponding eigenvectors as columns
    """
    eigenvals, eigenvectors = eigh(S)

    sort_indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    total_variance = np.sum(eigenvals)
    explained_variance_ratio = eigenvals / total_variance
    selected_indices = np.where(explained_variance_ratio > prop)[0] # Select indices meeting proportion

    if len(selected_indices) == 0:
        selected_indices = np.array([0])

    Lambda = np.diag(eigenvals[selected_indices]) # Diagonal matrix of selected eigenvalues
    U = eigenvectors[:, selected_indices]

    return Lambda, U


def project_and_reconstruct_image(image, U):
    """
    Compute the PCA representation of the image and reconstruct the image.
    :param image: original image
    :param U: the matrix of top eigenvectors
    :return: reconstructed image
    """
    alpha = np.dot(U.T, image)

    # Reconstruct the image from the projection
    reconstructed_image = np.dot(U, alpha)

    return reconstructed_image


def display_image(im_orig_fullres, im_orig, im_reconstructed):
    """
    Visualizes the original high-res image, original image and reconstructed image
    :param im_orig_fullres: Original high-res image
    :param im_orig: Original image
    :param im_reconstructed: Reconstructed image
    :return: a matplotlib figure and three axes objects
    """
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
    fig.tight_layout()

    ax1.set_title('Original High Res')
    ax1.imshow(np.reshape(im_orig_fullres, (218, 178, 3)))
    ax1.axis('off')

    ax2.set_title('Original')
    ax2.imshow(np.reshape(im_orig, (60, 50)), cmap='gray', aspect='equal')
    ax2.axis('off')

    ax3.set_title('Reconstructed')
    ax3.imshow(np.reshape(im_orig, (60, 50)), cmap='gray', aspect='equal')
    ax3.axis('off')

    # Add colorbar for original and reconstructed image
    fig.colorbar(ax2.imshow(np.reshape(im_orig, (60, 50)), cmap='gray'), ax=ax2)
    fig.colorbar(ax3.imshow(np.reshape(im_reconstructed, (60, 50)), cmap='gray'), ax=ax3)

    return fig, ax1, ax2, ax3


def perturb_image(image, U, sigma):
    """
    Reconstruct image with perturbed weights
    :param image: Original image
    :param U: matrix of top eigenvectors
    :param sigma: Standard deviation
    :return: Reconstructed image after being perturbed
    """
    alpha = np.dot(U.T, image)
    perturbed_image = np.random.normal(0, sigma, size=alpha.shape)  # Gaussian noise
    alpha_perturbed = alpha + perturbed_image

    return np.dot(U, alpha_perturbed)
