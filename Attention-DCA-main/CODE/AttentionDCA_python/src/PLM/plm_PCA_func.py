import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

    # One-hot encode
    one_hot_encoded = one_hot_seq_batch(sequences, max_pot=max_pot)

    # Flatten and scale
    flat = one_hot_encoded.reshape(one_hot_encoded.shape[0], -1)
    scaled = StandardScaler().fit_transform(flat)

    # PCA
    pca_result = PCA(n_components=2).fit_transform(scaled)
    plt.figure(figsize=(7, 6))
    if not (comparison_data is None):
        one_hot_encoded_test_data = one_hot_seq_batch(comparison_data, max_pot=max_pot)

        # Flatten and scale
        flat_data_test = one_hot_encoded_test_data.reshape(one_hot_encoded_test_data.shape[0], -1)
        scaled_data_test = StandardScaler().fit_transform(flat_data_test)

        # PCA
        pca_result_data_test = PCA(n_components=2).fit_transform(scaled_data_test)
        plt.scatter(pca_result_data_test[:, 0], pca_result_data_test[:, 1], alpha=0.5, s=10,label='Test Data')
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