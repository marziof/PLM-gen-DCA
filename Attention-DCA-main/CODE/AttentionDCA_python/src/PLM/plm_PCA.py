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
def plot_pca_of_sequences(sequences, title="PCA of Sequences", max_pot=21, save_path=None):
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

    # Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
##############################################################
"""
    One-hot encode AA sequences - plm generated and true
"""
filename = 'gen_seqs_w_init_seq_Ns40000_r0.1'
#filename = 'generated_sequences_randinit_40000'
#filename = 'generated_sequences_10000'
#cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main'
cwd='C:\Users\youss\OneDrive\Bureau\master epfl\MA2\TP4 De los Rios\git_test\PLM-gen-DCA\Attention-DCA-main'
# Load the generated sequences
#output_file = cwd + f'/CODE/AttentionDCA_python/src/PLM/generated_sequences/{filename}.npy'

output_file = cwd + f'\CODE\AttentionDCA_python\src\PLM\generated_sequences\{filename}.npy'
gen_sequences = np.load(output_file)
saved_seq = gen_sequences.copy()
gen_sequences = gen_sequences[10000:35000]

# Load train sequences
family = 'jdoms_bacteria_train2'
#filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'

filename = cwd + f'\CODE\DataAttentionDCA\jdoms\{family}.fasta'

train_sequences = sequences_from_fasta(filename)
train_sequences_num = [letters_to_nums(seq) for seq in train_sequences]

# One-hot encode the sequences
train_sequences_one_hot = one_hot_seq_batch(train_sequences_num, max_pot=21)
gen_sequences_one_hot = one_hot_seq_batch(gen_sequences, max_pot=21)

# Print out shapes for sanity check
print("True seqs one-hot shape:", train_sequences_one_hot.shape)
print("Gen  seqs one-hot shape:", gen_sequences_one_hot.shape)

##############################################################
"""
    PCA procedure
"""
# 1. Flatten
gen_flat = gen_sequences_one_hot.reshape(gen_sequences_one_hot.shape[0], -1)
train_flat = train_sequences_one_hot.reshape(train_sequences_one_hot.shape[0], -1)

# 2. Scale separately
scaler_train = StandardScaler()
train_scaled = scaler_train.fit_transform(train_flat)

scaler_gen = StandardScaler()
gen_scaled = scaler_gen.fit_transform(gen_flat)

# 3. PCA separately
pca_train = PCA(n_components=2)
train_pca = pca_train.fit_transform(train_scaled)

pca_gen = PCA(n_components=2)
gen_pca = pca_gen.fit_transform(gen_scaled)

##############################################################
"""
    PCA plots
"""
# Side-by-side PCA plots
plt.figure(figsize=(12, 5))
# Left plot: PCA of True Sequences
plt.subplot(1, 2, 1)
plt.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.5, s=10)
plt.title("PCA of True Sequences")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
# Right plot: PCA of Generated Sequences
plt.subplot(1, 2, 2)
plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, s=10, color='orange')
plt.title("PCA of Generated Sequences")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
# Adjust layout to avoid overlap
plt.tight_layout()
# Save the side-by-side PCA plot
save_path = os.path.join(cwd, 'CODE', 'AttentionDCA_python', 'src', 'PLM', 'results', 'PCA_plots')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Save PCA plot side-by-side
plt.savefig(save_path + 'PCA_plots.png')
print("Saved PCA plots to:", save_path + 'PCA_plots.png')
# After saving, display the plot
plt.show()

# PCA plot both in one graph
plt.figure(figsize=(8, 6))
# Plot both true and generated sequences on the same graph
plt.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.5, s=10, label='True Sequences')
plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, s=10, color='orange', label='Generated Sequences')
plt.title("PCA of True and Generated Sequences")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
# Save the combined PCA plot
plt.savefig(save_path + 'PCA_plots_onefig.png')
# Display the combined PCA plot
plt.show()