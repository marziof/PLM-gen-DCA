import numpy as np

from plm_seq_reader import sequences_from_fasta, letters_to_nums, nums_to_letters
from plm_model import SequencePLM

path = "/Users/marzioformica/Desktop/EPFL/Master/MA2/Labo/Attention-DCA-main/results/PLM/plm_generated_sequences.npy"
gen_sequences = np.load(path)
print("Generated sequences shape: ", gen_sequences.shape)

sequence = gen_sequences[10]

print("Generated sequence: ", sequence)
J = np.random.randn(10, 10, 63)
print("Generated sequence letters: ", SequencePLM(J,sequence).to_letter())
print("Generated sequence (method): ", nums_to_letters(sequence))