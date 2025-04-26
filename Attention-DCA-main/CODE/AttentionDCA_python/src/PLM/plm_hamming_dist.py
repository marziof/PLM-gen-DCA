
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

#Correlation

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
