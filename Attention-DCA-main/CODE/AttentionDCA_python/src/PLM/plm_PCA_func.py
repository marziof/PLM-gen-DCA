import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt


import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from dcascore import *
# back to original path (in PLM)
sys.path.pop(0)  # Removes the parent_dir from sys.path
from plm_seq_utils import letters_to_nums, sequences_from_fasta, one_hot_seq_batch


matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
font = {'size'   : 18}

matplotlib.rc('font', **font)

############### PCA function #################################
def plot_pca_of_sequences(sequences, title="PCA of Sequences",comparison_data=None ,max_pot=21, save_path=None,pca_graph_restrict=True):
    """
    Plots PCA of a list of sequences (strings or numerical) after one-hot encoding.

    Parameters:
    - sequences: list of sequences (strings or integer lists)
    - title: title of the PCA plot
    - max_pot: number of possible categories for one-hot encoding (default: 21)
    - save_path: optional path to save the plot
    """

    # Convert to numerical if needed
    if isinstance(sequences[0], str):
        sequences = [letters_to_nums(seq) for seq in sequences]

        
    plt.figure(figsize=(7, 6))
    if not (comparison_data is None):
        one_hot_encoded_test_data = one_hot_seq_batch(comparison_data, max_pot=max_pot)

        # Flatten and scale
        flat_data_test = one_hot_encoded_test_data.reshape(one_hot_encoded_test_data.shape[0], -1)
        scaler_data=StandardScaler()
        scaled_data_test = scaler_data.fit_transform(flat_data_test)

        # PCA
        pca_data=PCA(n_components=2)
        pca_result_data_test = pca_data.fit_transform(scaled_data_test)
        plt.scatter(pca_result_data_test[:, 0], pca_result_data_test[:, 1], alpha=0.5, s=10,label='Test Data')
    # One-hot encode
    one_hot_encoded = one_hot_seq_batch(sequences, max_pot=max_pot)

    # Flatten and scale
    flat = one_hot_encoded.reshape(one_hot_encoded.shape[0], -1)
    scaled = scaler_data.transform(flat)

    # PCA
    pca_result = pca_data.transform(scaled)
    
    
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10,label='Sequence Data')

    # Plot
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    if pca_graph_restrict and not (comparison_data is None):
        plt.xlim(1.5*np.min(pca_result_data_test[:, 0]),1.5*np.max(pca_result_data_test[:, 0]))
        plt.ylim(1.5*np.min(pca_result_data_test[:, 1]),1.5*np.max(pca_result_data_test[:, 1]))

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_projected_pca(sequences_reference, sequences_to_project, 
                       title="PCA: Reference vs Projected Sequences", 
                       max_pot=21, save_path=None, restrict_axes=True):
    """
    Projects `sequences_to_project` into the PCA space of `sequences_reference` and plots the PCA.

    Parameters:
    - sequences_reference: list of reference sequences (strings or integer lists)
    - sequences_to_project: list of sequences to project (strings or integer lists)
    - title: title of the PCA plot
    - max_pot: number of possible categories for one-hot encoding (default: 21)
    - save_path: optional path to save the plot
    - restrict_axes: restrict axes limits based on reference PCA
    """

    # Convert to numerical if needed
    if isinstance(sequences_reference[0], str):
        sequences_reference = [letters_to_nums(seq) for seq in sequences_reference]
    if isinstance(sequences_to_project[0], str):
        sequences_to_project = [letters_to_nums(seq) for seq in sequences_to_project]

    # One-hot encode
    one_hot_ref = one_hot_seq_batch(sequences_reference, max_pot=max_pot)
    one_hot_proj = one_hot_seq_batch(sequences_to_project, max_pot=max_pot)

    # Flatten
    ref_flat = one_hot_ref.reshape(one_hot_ref.shape[0], -1)
    proj_flat = one_hot_proj.reshape(one_hot_proj.shape[0], -1)

    # Scale using reference stats
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_flat)
    proj_scaled = scaler.transform(proj_flat)

    # PCA on reference only
    pca = PCA(n_components=2)
    ref_pca = pca.fit_transform(ref_scaled)
    proj_pca = pca.transform(proj_scaled)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, s=10, label='Reference Sequences')
    plt.scatter(proj_pca[:, 0], proj_pca[:, 1], alpha=0.5, s=10, color='orange', label='Projected Sequences')

    #plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    if restrict_axes:
        x_margin = 0.1 * (ref_pca[:, 0].max() - ref_pca[:, 0].min())
        y_margin = 0.1 * (ref_pca[:, 1].max() - ref_pca[:, 1].min())
        plt.xlim(ref_pca[:, 0].min() - x_margin, ref_pca[:, 0].max() + x_margin)
        plt.ylim(ref_pca[:, 1].min() - y_margin, ref_pca[:, 1].max() + y_margin)

    if save_path:
        plt.savefig(save_path)
    plt.show()


from scipy.stats import gaussian_kde


def plot_projected_pca_colormap(sequences_reference, sequences_to_project, 
                                title="PCA: Reference vs Projected Sequences (Colormap)", 
                                max_pot=21, save_path=None, restrict_axes=True,
                                cmap_ref='viridis', cmap_proj='plasma'):
    """
    Projects `sequences_to_project` into the PCA space of `sequences_reference` and plots both with KDE-based colormaps.

    Parameters:
    - sequences_reference: list of reference sequences (strings or integer lists)
    - sequences_to_project: list of sequences to project (strings or integer lists)
    - title: title of the PCA plot
    - max_pot: number of possible categories for one-hot encoding
    - save_path: optional path to save the plot
    - restrict_axes: restrict axes limits based on reference PCA
    - cmap_ref: colormap for reference sequences
    - cmap_proj: colormap for projected sequences
    """

    # Convert to numerical if needed
    if isinstance(sequences_reference[0], str):
        sequences_reference = [letters_to_nums(seq) for seq in sequences_reference]
    if isinstance(sequences_to_project[0], str):
        sequences_to_project = [letters_to_nums(seq) for seq in sequences_to_project]

    # One-hot encode
    one_hot_ref = one_hot_seq_batch(sequences_reference, max_pot=max_pot)
    one_hot_proj = one_hot_seq_batch(sequences_to_project, max_pot=max_pot)

    # Flatten and scale using reference stats
    ref_flat = one_hot_ref.reshape(one_hot_ref.shape[0], -1)
    proj_flat = one_hot_proj.reshape(one_hot_proj.shape[0], -1)

    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_flat)
    proj_scaled = scaler.transform(proj_flat)

    # PCA on reference
    pca = PCA(n_components=2)
    ref_pca = pca.fit_transform(ref_scaled)
    proj_pca = pca.transform(proj_scaled)

    # Compute densities for coloring
    ref_kde = gaussian_kde(ref_pca.T)
    proj_kde = gaussian_kde(proj_pca.T)
    ref_density = ref_kde(ref_pca.T)
    proj_density = proj_kde(proj_pca.T)

    # Axis limits for consistency
    if restrict_axes:
        x_min, x_max = ref_pca[:, 0].min(), ref_pca[:, 0].max()
        y_min, y_max = ref_pca[:, 1].min(), ref_pca[:, 1].max()
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)
        xlim = (x_min - x_margin, x_max + x_margin)
        ylim = (y_min - y_margin, y_max + y_margin)
    else:
        xlim = ylim = None
    # Plot side-by-side
    plt.figure(figsize=(14, 6))

    # Left: Reference
    plt.subplot(1, 2, 1)
    plt.scatter(ref_pca[:, 0], ref_pca[:, 1], c=ref_density, cmap=cmap_ref, s=10)
    plt.title("PCA of Reference Sequences")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    # Right: Projected
    plt.subplot(1, 2, 2)
    plt.scatter(proj_pca[:, 0], proj_pca[:, 1], c=proj_density, cmap=cmap_proj, s=10)
    plt.title("PCA of Projected Sequences")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_2d_histogram(data, bins=50, range_=None):
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=range_)
    hist = hist + 1e-10  # add small constant to avoid division by zero
    hist /= np.sum(hist)  # normalize to get probability distribution
    return hist

def compute_and_plot_2d_histogram(data, bins=50, range_=None, cmap='viridis'):
    # Compute histogram
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=range_)
    hist = hist + 1e-10  # avoid zero for stability
    hist /= np.sum(hist)  # normalize to get probability distribution

    # Plot heatmap
    plt.figure(figsize=(8,6))
    # Extent for imshow: left, right, bottom, top
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
    plt.colorbar(label='Probability density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Histogram (Normalized)')
    plt.show()

    return hist

from scipy.stats import entropy

def compute_kl_divergence_prev(p, q):
    return entropy(p.flatten(), q.flatten())

def compute_kl_divergence(P, Q, epsilon=1e-22):
    P = P + epsilon
    Q = Q + epsilon
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    return np.sum(P * np.log(P / Q))

def kl_divergence_between_pca_distributions(pca_data_1, pca_data_2, bins=50):
    # Define a shared range for both histograms
    combined = np.vstack((pca_data_1, pca_data_2))
    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    range_ = [[x_min, x_max], [y_min, y_max]]

    # Compute histograms (probability distributions)
    p = compute_2d_histogram(pca_data_1, bins=bins, range_=range_)
    q = compute_2d_histogram(pca_data_2, bins=bins, range_=range_)
    diff = p-q
    print("Max diff: ", np.max(diff))
    # KL divergence
    kl_pq = compute_kl_divergence(p, q)
    kl_qp = compute_kl_divergence(q, p)

    return kl_pq, kl_qp

def return_pca_results(sequences,comparison_data,max_pot=21):
    if isinstance(sequences[0], str):
        sequences = [letters_to_nums(seq) for seq in sequences]

        
    
    
    one_hot_encoded_test_data = one_hot_seq_batch(comparison_data, max_pot=max_pot)

    # Flatten and scale
    flat_data_test = one_hot_encoded_test_data.reshape(one_hot_encoded_test_data.shape[0], -1)
    scaler_data=StandardScaler()
    scaled_data_test = scaler_data.fit_transform(flat_data_test)

    # PCA
    pca_data=PCA(n_components=2)
    pca_result_data_test = pca_data.fit_transform(scaled_data_test)
    #plt.scatter(pca_result_data_test[:, 0], pca_result_data_test[:, 1], alpha=0.5, s=10,label='Test Data')
# One-hot encode
    one_hot_encoded = one_hot_seq_batch(sequences, max_pot=max_pot)

    # Flatten and scale
    flat = one_hot_encoded.reshape(one_hot_encoded.shape[0], -1)
    scaled = scaler_data.transform(flat)

    # PCA
    pca_result = pca_data.transform(scaled)
    return pca_result,pca_result_data_test

from scipy.spatial import cKDTree
import numpy as np

def average_minimum_distance(pca_true, pca_gen):
    tree_gen = cKDTree(pca_gen)
    dists_true_to_gen, _ = tree_gen.query(pca_true, k=1)
    return np.mean(dists_true_to_gen)

def symmetric_average_minimum_distance(pca_true, pca_gen):
    amd_true_to_gen = average_minimum_distance(pca_true, pca_gen)
    amd_gen_to_true = average_minimum_distance(pca_gen, pca_true)
    return 0.5 * (amd_true_to_gen + amd_gen_to_true)
