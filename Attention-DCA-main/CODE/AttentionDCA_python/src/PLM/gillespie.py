from tqdm import tqdm
import numpy as np

from plm_seq_utils import letter_to_num

class SequenceGill:
    # Logic: proba to assing a specific AA to a specific site = proba to choose site (uniform distrib) * proba to draw AA at site
    def __init__(self, J, initial_sequence = None, beta = 1,time=0):
        """
        Initialize the SequenceGill object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J
        self.L = J.shape[-1]
        self.beta = beta
        self.time=0
        if initial_sequence is None:
            self.sequence = np.random.choice(np.arange(21), self.L) # Sequence of ints (1 to 21)
        else:
            self.sequence = initial_sequence

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
            sum_energy += self.J[trial_aa, aa_j, site, j] # check indexing
            #sum_energy += self.J[aa_j, trial_aa, j, site]
        prob = np.exp(self.beta * sum_energy)  # unnormalized
        return prob
    
    def plm_site_distribution(self, site):
        """
        Compute probability distriution for specific site 
        """
        probs = []
        for trial_aa in range(21):
            if trial_aa==self.sequence[site]:
                probs.append(0)
            else:
                probs.append(self.plm_calc(site, trial_aa))
        probs = np.array(probs)
        return probs
    
    def gillespie_seq(self):
        """
        return matrix of probabilities of shape (L,21)
        """
        probs=[]
        for site in range(self.L):
            probs.append(self.plm_site_distribution(site))
        probs= np.array(probs)
        return probs
    
    def draw_aa(self):
        """
        Sample a new AA at the given site from PLM distribution
        """
        probs = self.gillespie_seq()
        k_hat=probs.sum()
        probs=probs/k_hat
        flat_probs=probs.ravel()
        rows ,cols = probs.shape
        sampled_index = np.random.choice(flat_probs.size, p=flat_probs)
        site, new_aa = np.unravel_index(sampled_index, (rows, cols))
        self.sequence[site] = new_aa
        self.time=np.random.exponential(1/k_hat)

    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum