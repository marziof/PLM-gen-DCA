
# Read your sequences from a txt file and output a fasta file

<<<<<<< HEAD
seq_dir = 'mc_generated_sequences'
#filename = 'generated_sequences_randinit_40000'
filename = 'mc_gen_seqs_w_init_seq_Ns300000_r0.1'
=======

seq_dir = 'generated_sequences'
#filename = 'generated_sequences_randinit_40000'
filename = 'gen_seqs_w_init_seq_Ns40000_r0.1'
#seq_dir = 'mc_generated_sequences'
#filename = 'mc_gen_seqs_w_init_seq_Ns300000_r0.1'
#filename = 'mc_generated_sequences_randinit_300000'
>>>>>>> 753d3cd4c21be9074c0145f018e838ee4c2ae51c
txt_file = f'{seq_dir}/{filename}.txt'

# Read all sequences first
with open(txt_file, 'r') as infile:
    sequences = [line.strip().replace('-', '') for line in infile if line.strip()]

# Make sure there are enough sequences
if len(sequences) <= 5000:
    raise ValueError("Not enough sequences to start at index 5000!")

# Select sequences starting from index 5000
selected_sequences = sequences[5000:]

# Pick 100 evenly spaced sequences
import numpy as np
indices = np.linspace(0, len(selected_sequences) - 1, 100, dtype=int)
evenly_spaced_sequences = [selected_sequences[i] for i in indices]

# Now write to fasta
with open(f'{seq_dir}/{filename}.fasta', 'w') as outfile:
    for idx, seq in enumerate(evenly_spaced_sequences):
        outfile.write(f">seq{5000 + indices[idx] + 1}\n{seq}\n")  # adjust seq id based on original file