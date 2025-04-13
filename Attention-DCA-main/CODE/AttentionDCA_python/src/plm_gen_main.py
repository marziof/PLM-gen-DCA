import numpy as np
import torch
from model import AttentionModel
from dcascore import *
import os
import random

from plm_model import SequencePLM
from plm_model import generate_plm

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
"""
    Generate sequences with PLM
"""
gen_sequences = []
#for i in range(100):
 #   print("sequence number: ", i)
  #  seq = generate_plm(Jtens, 640)
   # gen_sequences.append(seq)
gen_sequences = generate_plm(Jtens, 10000)

# File path
output_file = cwd + '/results/PLM/plm_generated_sequences.npy'

# Save using NumPy's save function
np.save(output_file, gen_sequences)

## Save the generated sequences to a file
#if not os.path.exists(cwd + '/results/PLM'):
#    os.makedirs(cwd + '/results/PLM')
#output_file = cwd + '/results/PLM/plm_generated_sequences.txt'
#with open(output_file, 'w') as f:
#    for sequence in gen_sequences:
#        f.write(''.join(str(x) for x in sequence) + '\n')
##############################################################