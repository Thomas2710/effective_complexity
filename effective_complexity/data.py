import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import os
from datetime import datetime

matplotlib.use('Qt5Agg')  # Use an interactive backend like TkAgg

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

def show_distrib(distrib, method = 'NOT', predicted = True, show = False, experiment = ('SYNTHETIC', 'MLP')):

    dim = distrib.shape[1]
    if predicted:
        exp = 'predicted'
    else:
        exp = 'reference'
    # Plotting 2D data
    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(distrib[:, 0], distrib[:, 1], c='blue', alpha=0.6)
        plt.title(f"{method.upper()} Reduced Visualization of Data (2D)") 
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    
    # Plotting 3D data
    elif dim == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(distrib[:, 0], distrib[:, 1], distrib[:, 2], c='blue', alpha=0.6)
        ax.set_title(f"{method.upper()} Reduced Visualization of Data (3D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
    else:
        raise ValueError("target_dim must be 2 or 3.")
    
    # Create the folder
    # Get the current time formatted as HH-MM-SS
    folder_name = ''+experiment[0]+'_'+experiment[1]+'_'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_path = os.path.join(os.getcwd(), 'PLOTS', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, ''+exp+'_'+str(dim)+'dim_'+method+'.png'))
    if show:
        plt.show()
    plt.close()


def apply_pca(data):
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    return pca_result

def apply_tsne(data):
    # Apply t-SNE to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data)
    return tsne_result
