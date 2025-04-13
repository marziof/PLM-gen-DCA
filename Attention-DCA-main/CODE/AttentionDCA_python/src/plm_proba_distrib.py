import numpy as np
import torch
from model import AttentionModel
from dcascore import *
import os
import random
import matplotlib.pyplot as plt

from plm_model import SequencePLM
from plm_model import generate_plm
from plm_seq_reader import letter_to_num, num_to_letter, sequences_from_fasta, letters_to_nums

##############################################################
"""
    Load Q, K, V matrices from jdoms (after training)
"""

def read_tensor_from_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Read the dimensions from the first line
    dims = list(map(int, lines[0].strip().split()))
    
    tensor_data = []
    current_slice = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Slice"):
            if current_slice:
                tensor_data.append(current_slice)
                current_slice = []
        elif line:
            current_slice.append(list(map(float, line.split(','))))
    if current_slice:
        tensor_data.append(current_slice)

    tensor = torch.tensor(tensor_data).view(*dims)
    return tensor

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# params of training:
H = 64
d= 10
N = 174
n_epochs = 500
loss_type = 'without_J'
family = 'jdoms_bacteria_train2'
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/Attention-DCA-main/' # path to Attention-DCA-main/
Q_1 = read_tensor_from_txt( cwd +"/results/MARZIO/{H}_{d}_{family}_{losstype}_{n_epochs}/K_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
K_1 = read_tensor_from_txt( cwd +"/results/MARZIO/{H}_{d}_{family}_{losstype}_{n_epochs}/Q_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))
V_1 = read_tensor_from_txt( cwd +"/results/MARZIO/{H}_{d}_{family}_{losstype}_{n_epochs}/V_tensor.txt".format(H=H, d=d, family=family, losstype=loss_type, n_epochs=n_epochs))

H,d,N=Q_1.shape
q=V_1.shape[1]

###
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'
structfile = cwd + '/CODE/DataAttentionDCA/jdoms/jdom.dat'
file_path = filename

def read_fasta(file_path):
    with open(file_path, "r") as file:
        for line in file:
            print(line.strip())  # Strip removes extra whitespace

##############################################################
"""
    Initialize the model and compute couplings J from Q, K, V
""" 
model=AttentionModel(H,d,N,q,Q_1,K_1,V_1) # Check version of AttentionModel (updated to receive Q,K,V so it can be initialized without training)
device = Q_1.device
L = Q_1.shape[-1]
W=attention_heads_from_model(model,Q_1,K_1,V_1)
print(W.shape)

i_indices = torch.arange(L, device=device).unsqueeze(1)
j_indices = torch.arange(L, device=device).unsqueeze(0)
mask = (i_indices != j_indices).float().unsqueeze(0)  # shape (1, L, L)
W = W * mask
    
# Compute Jtens
Jtens = torch.einsum('hri,hab->abri', W, V_1)  # Shape: (q, q, L, L)
q = Jtens.shape[0]
N = Jtens.shape[2]

##############################################################
#TESTS
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/Attention-DCA-main/' # path to Attention-DCA-main/
save_path = cwd + '/results/PLM/Proba_plots/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

"""
    Test 1: start from random sequence
"""

seq = SequencePLM(Jtens,sequence=None)
print(seq.sequence)
print("Initial seq:", seq.to_letter())
site = np.random.randint(seq.L)  # Random site from 0 to L-1
print("site:", site)  

N = 100
drawn_aas = []
for i in range(N):
    seq.draw_aa(site)
    aa = seq.sequence[site]
    drawn_aas.append(aa)
# Count occurrences of each amino acid ID (1 to 21)
aa_ids = np.arange(1, 22)
frequencies = [drawn_aas.count(i) for i in aa_ids]  # Pure Python count

# Get empirical frequencies
aa_ids = np.arange(1, 22)
empirical_freqs = [drawn_aas.count(i) / len(drawn_aas) for i in aa_ids]  # Normalize to probability

# Get PLM probabilities at the site
plm_probs = seq.plm_site_distribution(site)
# Plot both
plt.figure(figsize=(10, 6))
bar_width = 0.4

# Bar plot for empirical frequencies
print(plm_probs.shape)
print(empirical_freqs.shape)
plt.bar(aa_ids - bar_width/2, empirical_freqs, width=bar_width, label='Drawn Frequency', alpha=0.7)
print("first ok")
# Bar plot for PLM distribution
plt.bar(aa_ids + bar_width/2, plm_probs, width=bar_width, label='PLM Probability', alpha=0.7)

# X-axis labels
plt.xticks(aa_ids, [num_to_letter[i] for i in aa_ids])
plt.xlabel('Amino Acid')
plt.ylabel('Probability')
plt.title(f'Empirical vs PLM Distribution at Site {site}')
plt.legend()
plt.tight_layout()
plt.savefig(save_path + 'Random_init.pdf')
plt.show()

###############################

"""
    Test 2: start from true sequence
"""

fasta_path = cwd + '/CODE/DataAttentionDCA/jdoms/jdoms_bacteria_train2.fasta'
true_seq = sequences_from_fasta(fasta_path)[0]  # Read the first sequence from the FASTA file
# Convert sequence to numerical representation
true_seq_nb = letters_to_nums(true_seq)
print("Original sequence:", true_seq)
print("Numerical sequence:", true_seq_nb)

# Select a random site
site = np.random.randint(seq.L)  # Random site from 0 to L-1
site = 28
print("Random site selected:", site, "with amino acid (letter):", num_to_letter[true_seq_nb[site]])

# Initialize the SequencePLM object and set the sequence
seq = SequencePLM(Jtens)
seq.sequence = true_seq_nb.copy()

print("Initial seq:", seq.to_letter())
print("True amino acid at site:", true_seq[site])

# Sampling new amino acids at the given site
N = 100
drawn_aas = []
drawn_sequences = []
original_seq_count = 0  # To count how many times the original sequence is drawn

# Sampling loop
for i in range(N):
    seq.draw_aa(site)  # Perform the draw
    aa = seq.sequence[site]  # Get the amino acid at the chosen site after the draw
    drawn_aas.append(aa)  # Store the drawn amino acid
    drawn_sequences.append(seq.sequence.copy())  # Store the full sequence
    
    # Check if the drawn sequence matches the original sequence
    if np.array_equal(seq.sequence, true_seq_nb):
        original_seq_count += 1

# Print the number of times the original sequence was drawn
print(f"The original sequence was drawn {original_seq_count} times out of {N}.")
# Calculate empirical frequencies (normalize to get probabilities)
aa_ids = np.arange(1, 22)
empirical_freqs = [drawn_aas.count(i) / len(drawn_aas) for i in aa_ids]  # Normalize to probability

# Get PLM probabilities at the site (1 to 21, excluding 0 index)
plm_probs = seq.plm_site_distribution(site)

# Plot both empirical frequencies and PLM distribution
plt.figure(figsize=(10, 6))
bar_width = 0.4

# Bar plot for empirical frequencies
plt.bar(aa_ids - bar_width / 2, empirical_freqs, width=bar_width, label='Drawn Frequency', alpha=0.7)

# Bar plot for PLM distribution
plt.bar(aa_ids + bar_width / 2, plm_probs, width=bar_width, label='PLM Probability', alpha=0.7)

# X-axis labels
plt.xticks(aa_ids, [num_to_letter[i] for i in aa_ids])
plt.xlabel('Amino Acid')
plt.ylabel('Probability')
plt.title(f'Empirical vs PLM Distribution at Site {site}')
plt.legend()
plt.tight_layout()
plt.savefig(save_path + 'True_init.pdf')
plt.show()