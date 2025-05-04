
from plm_seq_utils import letters_to_nums, sequences_from_fasta
#from plm_PCA import one_hot_seq_batch
import numpy as np
import matplotlib.pyplot as plt
import os


# ------------------------------- FUNCTIONS -------------------------------
def hamming_dist(seq1, seq2):
    """
    Calculate the Hamming distance between two sequences.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return np.sum(seq1 != seq2)

def hamming_dist_batch(sequences1, sequences2):
    """
    Calculate the Hamming distance between two batches of sequences.
    """
    if len(sequences1) != len(sequences2):
        # cut to keep the same number of sequences

        raise ValueError("Both batches must have the same number of sequences")
    
    distances = []
    for seq1, seq2 in zip(sequences1, sequences2):
        distances.append(hamming_dist(seq1, seq2))
    
    return np.array(distances)

def vectorized_hamming_distance(sequences1, sequences2):
    # Ensure the number of sequences are the same by truncating the longer array
    min_len = min(sequences1.shape[0], sequences2.shape[0])
    sequences1 = sequences1[:min_len].copy()
    sequences2 = sequences2[:min_len].copy()
    # Compare the sequences element-wise to count differences (direct numeric comparison)
    return np.sum(sequences1 != sequences2, axis=1)

# ------------------------------- MAIN -------------------------------

cwd = '/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main'
#cwd='C:\Users\youss\OneDrive\Bureau\master epfl\MA2\TP4 De los Rios\git_test\PLM-gen-DCA\Attention-DCA-main'

# --- Load sequences ---
filename = 'gen_seqs_w_init_seq_Ns40000_r0.1'
simu_name = 'init_seq_Ns40000_r0.1'
filename = 'generated_sequences_randinit_40000'
simu_name = 'randinit_Ns40000'
output_file = cwd + f'/CODE/AttentionDCA_python/src/PLM/generated_sequences/{filename}.npy'
gen_sequences=np.load(output_file)

family = 'jdoms_bacteria_train2'
filename = cwd + f'/CODE/DataAttentionDCA/jdoms/{family}.fasta'

# Get the raw letter sequences from the FASTA file
folder_name = "/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/my_project/PLM-gen-DCA/Attention-DCA-main/CODE/AttentionDCA_python/src/my_saved_data"
os.makedirs(folder_name, exist_ok=True)

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

# ----------------------- RESULTS -----------------------
# --- Create save directory ---
cwd = os.getcwd()
save_path = cwd + '/results/Hamming_distances'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#################################################################
### 1. Hamming distance between generated sequences and training sequences
# --- define subsets ---
# Define the index range
start_idx = 1
end_idx = 40000
step_size = 1  # Select every 1000th sequence

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
print("Saved plot to:", save_path + f'/Hd_GenTrain_{simu_name}.pdf')
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
print("Saved plot to:", save_path + f'/Hd_test_train_{simu_name}.pdf')
#plt.show()

# Print stats
print("Vectorized Hamming distances (within train pairs):", distance_within_train)
print("Average Hamming distance (within train pairs):", np.mean(distance_within_train))


#################################################################
### 2. Compare Hamming distances within generated sequences - evolution wrt initial sequence ---

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
plt.savefig(save_path + f'/Hd_gen_init_{simu_name}.pdf')
#plt.show()

# Print some statistics
print(f"Average Hamming distance: {np.mean(hamming_distances)}")

# Correlation:

def energy_corr_step(energy_seq,cor_step):
    avr_en=np.mean(energy_seq)
    first=energy_seq[:-cor_step]-avr_en
    second=energy_seq[cor_step:]-avr_en
    numerator=np.mean(first*second)
    denomin=np.sqrt(np.mean(first**2)*np.mean(second**2))
    return numerator/denomin

def energy_corr_array(energy_seq,max_cor_step):
    list_corr=[]
    for i in range(max_cor_step):
        list_corr.append(energy_corr_step(energy_seq,i+1))
    return np.array(list_corr)

corr_energy_plot=energy_corr_array(hamming_distances,int(len(hamming_distances)))
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
plt.savefig(save_path + f'/Hd_correlation_{simu_name}.pdf')
#plt.show()
