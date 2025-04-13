from tqdm import tqdm
import numpy as np

from plm_seq_reader import letter_to_num

class SequencePLM:
    def __init__(self, J, sequence = None):
        self.J = J
        self.L = J.shape[-1]
        if sequence is None:
            self.sequence = np.random.choice(np.arange(1, 22), self.L) # Sequence of ints (1 to 21)
        else:
            self.sequence = sequence

    def to_letter(self):
        """
        Show sequence as letters
        """
        print("Sequence:", self.sequence)
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        letter_seq = ''.join([num_to_letter[i] for i in self.sequence])
        print(letter_seq)
        return letter_seq

    def plm_calc(self, site, trial_aa):
        """
        Compute unnormalized pseudo-likelihood of trial_aa at a given site.
        
        site: int from 0 to L-1
        trial_aa: int from 0 to 21 (amino acid index)
        """
        sum_energy = 0.0
        for j in range(self.L):
            if j == site:
                continue
            aa_j = self.sequence[j]
            sum_energy += self.J[trial_aa-1, aa_j-1, site, j] # check indexing
        prob = np.exp(sum_energy)  # unnormalized
        return prob
    
    def plm_site_distribution(self, site):
        """
        Compute probability distriution for specific site (normalized)
        """
        probs = []
        for trial_aa in range(21):
            probs.append(self.plm_calc(site, trial_aa))
        probs = np.array(probs)
        probs /= probs.sum()
        return probs
    
    def draw_aa(self, site):
        """
        Sample a new AA at the given site from PLM distribution
        """
        probs = self.plm_site_distribution(site)
        new_aa = np.random.choice(22, p=probs)
        self.sequence[site] = new_aa


def generate_plm(J,maxiter=10000):
    """
    function to generate new sequence using PLM
    """
    gen_sequences = []
    seq = SequencePLM(J)
    for _ in tqdm(range(maxiter)):
        site = np.random.randint(seq.L)  # Random site from 0 to L-1
        seq.draw_aa(site)
        gen_sequences.append(seq.sequence.copy())
    gen_sequences = np.array(gen_sequences)
    return gen_sequences