from plm_seq_utils import letters_to_nums, sequences_from_fasta
#from plm_PCA import one_hot_seq_batch
import numpy as np
import matplotlib.pyplot as plt
import os
from plm_hamming_dist import *







# MAIN

# --- Load sequences ---
# Define the file path
#filename = 'generated_sequences_40000'
filename = 'generated_sequence_randinit_40000'
cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main'
output_file = cwd + f'/CODE/AttentionDCA_python/src/PLM/generated_sequences/{filename}.npy'
family = 'jdoms_bacteria_train2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'

# Get the raw letter sequences from the FASTA file
folder_name = "/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main/CODE/AttentionDCA_python/src/my_saved_data"
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, "plm_generated_V21_gap_seqs_jdom_40000_exp_pos_init_mod.txt")
seq_aa=np.loadtxt(file_path,dtype=int)
gen_sequences = seq_aa
#gen_sequences = np.load(output_file)
train_sequences = sequences_from_fasta(filename)
# Convert to numeric sequences
train_sequences_num = np.array([letters_to_nums(seq) for seq in train_sequences])

# Test sequences
family = 'jdoms_bacteria_test2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'
# Initialize a list to store the sequences
test_sequences = sequences_from_fasta(filename)
test_sequences_num = np.array([letters_to_nums(seq) for seq in test_sequences])

print("train sequences: ", np.shape(train_sequences_num))
print("test sequences: ", np.shape(test_sequences_num))
print("gen sequences: ", np.shape(gen_sequences))

#################################################################
# Results
cwd = os.getcwd()
save_path = cwd + '/results/Hamming_distances'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- Select every 1000th sequence ---
# Define the index range
start_idx = 2000
end_idx = 40000
step_size = 100  # Select every 1000th sequence

# Select sequences between start and end, with step
gen_sequences_subset = gen_sequences[start_idx:end_idx:step_size]
train_sequences_num_subset = train_sequences_num[start_idx:end_idx:step_size]

# --- Calculate Hamming distances ---
distances = vectorized_hamming_distance(gen_sequences_subset, train_sequences_num_subset)

# --- Plot the results ---
# Plot Hamming distances as a function of sequence index
plt.figure(figsize=(8, 6))
plt.scatter(np.arange(0, len(distances) * step_size, step_size), distances, alpha=0.5)
plt.title("Hamming Distances as a Function of Sequence Index")
plt.xlabel("Sequence Index")
plt.ylabel("Hamming Distance")
plt.savefig(save_path + '/Hd_gen_train.pdf')
print("Saved plot to:", save_path + '/Hd_gen_train.pdf')
#plt.show()

# Print out some statistics
print("Vectorized Hamming distances (every f{step_size}th sequence):", distances)
print("Average Hamming distance (every 1000th sequence):", np.mean(distances))

distance_test_train = vectorized_hamming_distance(train_sequences_num, test_sequences_num)
print("Vectorized Hamming distances (train vs test):", distance_test_train)
print("Average Hamming distance (train vs test):", np.mean(distance_test_train))



# --- Compare Hamming distances within train sequences (first vs last, second vs second-last, etc.) ---
# Determine number of pairs (half the number of sequences)
num_pairs = train_sequences_num.shape[0] // 2

# Pair first with last, second with second last, etc.
train_seq_subset_1 = train_sequences_num[:num_pairs]
train_seq_subset_2 = train_sequences_num[-num_pairs:][::-1]  # Reverse last part

# Compute Hamming distances
distance_within_train = vectorized_hamming_distance(train_seq_subset_1, train_seq_subset_2)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(np.arange(num_pairs), distance_within_train, alpha=0.5)
plt.title("Hamming Distances Between Train Sequence Pairs")
plt.xlabel("Pair Index (first vs last, etc.)")
plt.ylabel("Hamming Distance")
plt.savefig(save_path + '/Hd_test_train.pdf')
print("Saved plot to:", save_path + '/Hd_test_train.pdf')
#plt.show()

# Print stats
print("Vectorized Hamming distances (within train pairs):", distance_within_train)
print("Average Hamming distance (within train pairs):", np.mean(distance_within_train))


#################################################################
# --- Compare Hamming distances within generated sequences - evolution wrt initial sequence ---

initial_sequence = gen_sequences[0]  # Example, should be the initial sequence you are working with

# Compute Hamming distances sequentially
hamming_distances = []
for seq in gen_sequences:
    distance = hamming_dist(initial_sequence, seq)
    hamming_distances.append(distance)

# Convert the list of distances into a numpy array
hamming_distances = np.array(hamming_distances)

# Plot Hamming distances as a function of sequence index
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(hamming_distances)), hamming_distances, alpha=0.5)
plt.title("Hamming Distances Between Initial Sequence and Generated Sequences")
plt.xlabel("Sequence Index")
plt.ylabel("Hamming Distance")
plt.savefig(save_path + '/Hd_gen_init.pdf')
print("Saved plot to:", save_path + '/Hd_gen_init.pdf')
#plt.show()

# Print some statistics
print(f"Average Hamming distance: {np.mean(hamming_distances)}")

# Correlation:



corr_energy_plot=energy_corr_array(hamming_distances,int(len(hamming_distances)/5))
# X-axis: correlation step (1 to max_cor_step)
x_vals = np.arange(1, len(corr_energy_plot) + 1)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, corr_energy_plot, marker='o', linestyle='-')
plt.title("Autocorrelation of Hamming Distance vs Sequence Index")
plt.xlabel("Correlation Step (sequence offset)")
plt.ylabel("Correlation")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path + '/Hd_gen_correlation.pdf')
print("Saved plot to:", save_path + '/Hd_gen_correlation.pdf')
#plt.show()