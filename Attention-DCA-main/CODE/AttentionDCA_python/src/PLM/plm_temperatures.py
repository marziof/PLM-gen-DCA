from plm_seq_utils import letters_to_nums, sequences_from_fasta
#from plm_PCA import one_hot_seq_batch
import numpy as np
import matplotlib.pyplot as plt
import os

from plm_hamming_dist import hamming_dist, energy_corr_array

#----------- Load sequences with different betas -------------
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main'
#cwd='C:\Users\youss\OneDrive\Bureau\master epfl\MA2\TP4 De los Rios\git_test\PLM-gen-DCA\Attention-DCA-main'

betas = [0.1, 0.01, 0.5, 1, 2, 4]
correlations = []
for b in betas:
    filename = f'gen_seqs_w_init_seq_Ns40000_r0.1_b{b}'
    output_file = cwd + f'/CODE/AttentionDCA_python/src/PLM/generated_sequences/{filename}.npy'
    gen_sequences = np.load(output_file)

    initial_sequence = gen_sequences[0]
    hamming_distances = []
    for seq in gen_sequences:
        distance = hamming_dist(initial_sequence, seq)
        hamming_distances.append(distance)
    hamming_distances = np.array(hamming_distances)
    correlation = np.mean(energy_corr_array(hamming_distances,int(len(hamming_distances)/5)))
    correlations.append(correlation)
correlations = np.array(correlations)

#------------- Plot correlations vs temperature -------------------

cwd = os.getcwd()
save_path = cwd + '/results/Temperatures'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(betas, correlations, marker='o', linestyle='-')
plt.title("Autocorrelation of Hamming Distance vs Sequence Index")
plt.xlabel("Betas")
plt.ylabel("Correlation")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path + f'/temperature_correlation.pdf')