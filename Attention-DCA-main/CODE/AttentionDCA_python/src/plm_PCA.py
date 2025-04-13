import numpy as np
from dcascore import *
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Bio import SeqIO


##############################################################
"""
    One-hot encode AA sequences - plm generated and true
"""

def one_hot_seq_batch(seqs, max_pot=21):
    def one_hot_aa(aa):
        zeros = np.zeros(max_pot)
        zeros[aa - 1] = 1  # assumes aa in 1–21
        return zeros

    return np.array([[one_hot_aa(aa) for aa in seq] for seq in seqs])

cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/Attention-DCA-main/' # path to Attention-DCA-main/
#output_file = cwd + '/results/PLM/plm_generated_sequences.txt'
# File path
output_file = cwd + '/results/PLM/plm_generated_sequences.npy'

# Load using NumPy's load function
gen_sequences = np.load(output_file)

## Check the shape of the loaded sequences
#print(f"Shape of loaded sequences: {gen_sequences.shape}")
## Recreate gen_sequences by reading the file
#with open(output_file, 'r') as f:
#    gen_sequences = [list(map(int, line.strip())) for line in f]
#
## Check the lengths of the sequences
#seq_lengths = [len(seq) for seq in gen_sequences]
#print(f"Sequence lengths: {seq_lengths}")
#
## Find the maximum sequence length (you can also set this to a specific value if desired)
#max_len = max(seq_lengths)
#print(f"Maximum sequence length: {max_len}")
#
## Pad sequences to the same length (optional: pad with 0 or any other value)
#gen_sequences_padded = [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in gen_sequences]
#
## Convert the list of sequences to a NumPy array
#gen_sequences = np.array(gen_sequences_padded)
#
## Check the shape of the loaded sequences
#print(f"Shape of sequences: {gen_sequences.shape}")


# Check the shape of the loaded sequences
#print(gen_sequences.shape)
gen_sequences_one_hot = one_hot_seq_batch(gen_sequences,max_pot=21)

# one-hot true sequences

# Define the file path
family = 'jdoms_bacteria_train2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'

letter_to_num = {
    'A': 1,  'B': 21, 'C': 2,  'D': 3,  'E': 4,
    'F': 5,  'G': 6,  'H': 7,  'I': 8,  'J': 21,
    'K': 9,  'L': 10, 'M': 11, 'N': 12, 'O': 21,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
    'U': 21, 'V': 18, 'W': 19, 'X': 21, 'Y': 20,
    '-': 21  # Gap symbol
}
# Initialize a list to store the sequences
true_sequences = []

# Parse the FASTA file using BioPython's SeqIO
for record in SeqIO.parse(filename, "fasta"):
    # Extract the sequence (and its header if needed)
    sequence = str(record.seq)  # convert to string
    numeric_seq = [letter_to_num[aa] for aa in sequence]
    true_sequences.append(numeric_seq)
    #print(f"Extracted sequence: {sequence[:50]}...")  # Display first 50 characters of each sequence

true_sequences = np.array(true_sequences)
true_sequences_one_hot = one_hot_seq_batch(true_sequences,max_pot=21)

##############################################################
"""
    PCA procedure
"""
# 1. Flatten
gen_flat = gen_sequences_one_hot.reshape(gen_sequences_one_hot.shape[0], -1)
true_flat = true_sequences_one_hot.reshape(true_sequences_one_hot.shape[0], -1)

# 2. Scale separately
scaler_true = StandardScaler()
true_scaled = scaler_true.fit_transform(true_flat)

scaler_gen = StandardScaler()
gen_scaled = scaler_gen.fit_transform(gen_flat)

# 3. PCA separately
pca_true = PCA(n_components=2)
true_pca = pca_true.fit_transform(true_scaled)

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
plt.scatter(true_pca[:, 0], true_pca[:, 1], alpha=0.5, s=10)
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
save_path = cwd + '/results/PLM/PCA_plots/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save PCA plot side-by-side
plt.savefig(save_path + 'PCA_plots.png')

# After saving, display the plot
plt.show()

# PCA plot both in one graph
plt.figure(figsize=(8, 6))

# Plot both true and generated sequences on the same graph
plt.scatter(true_pca[:, 0], true_pca[:, 1], alpha=0.5, s=10, label='True Sequences')
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