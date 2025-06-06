import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


matplotlib.use('Qt5Agg')  # Use an interactive backend like TkAgg

#Show syntethic gaussian distribution
def show_gaussian(samples, reference = True):
    if reference:
        x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]

        # Need an (N, 2) array of (x, y) pairs.
        xy = np.column_stack([x.flat, y.flat])

        mu = np.array([0.0, 0.0])

        sigma = np.array([.5, .5])
        covariance = np.diag(sigma**2)

        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        # Reshape back to a (30, 30) grid.
        z = z.reshape(x.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x,y,z)
        #ax.plot_wireframe(x,y,z)

    # Plot the sampled points
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, label='Sampled Points', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Gaussian Distribution Sampling and Distribution')
    plt.legend()
    plt.show()

#Plot distribution in input, in plt ax in input
def show_distrib(distrib, method = 'NO', predicted = True, ax = None, limit=None):
    min_limit, max_limit = limit
    dim = distrib.shape[1]
    if predicted:
        exp = 'predicted'
    else:
        exp = 'reference'

    if dim == 2:
        ax.scatter(distrib[:, 0], distrib[:, 1], c='blue', alpha=0.6)
        ax.set_title(f"{method.upper()} Reduced Visualization of {exp} Data (2D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xlim(min_limit[0],max_limit[0])
        ax.set_ylim(min_limit[1],max_limit[1])
    elif dim == 3:
        ax.scatter(distrib[:, 0], distrib[:, 1], distrib[:, 2], c='blue', alpha=0.6)
        ax.set_title(f"{method.upper()} Reduced Visualization of {exp} Data (3D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.set_xlim(min_limit[0],max_limit[0])
        ax.set_ylim(min_limit[1],max_limit[1])
        ax.set_zlim(min_limit[2],max_limit[2])
    else:
        raise ValueError("target_dim must be 2 or 3.")

#Find the range of two distributions in each dimension
def find_distribution_limits(distrib1, distrib2):
    # Find min and max per dimension (i.e., column)
    min_per_dimension = np.min(distrib1, axis=0)
    max_per_dimension = np.max(distrib2, axis=0)

    min_per_pred_dimension = np.min(distrib1, axis=0)
    max_per_pred_dimension = np.max(distrib2, axis=0)

    min_limit = np.minimum(min_per_dimension, min_per_pred_dimension)
    max_limit = np.maximum(max_per_dimension, max_per_pred_dimension)
    limit = (min_limit, max_limit)
    return limit

#Apply PCA dim reduction to distribution
def apply_pca(data, num_components=2):
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(data)
    data_reconstructed = pca.inverse_transform(pca_result)
    variance_explained = np.cumsum(pca.explained_variance_ratio_)

    return pca_result, data_reconstructed, variance_explained

#Apply t-SNE dim reduction to distribution
def apply_tsne(data, num_components=2, perplexity=30, random_state=42):
    # Apply t-SNE to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(data)
    return tsne_result
