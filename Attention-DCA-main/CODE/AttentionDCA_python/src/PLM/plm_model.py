from tqdm import tqdm
import numpy as np

from plm_seq_utils import letter_to_num

class SequencePLM:
    def __init__(self, J, initial_sequence = None, beta = 1):
        """
        Initialize the SequencePLM object with a coupling tensor J of the family and an optional initial sequence.
        """
        self.J = J
        self.L = J.shape[-1]
        self.beta = beta
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
        new_aa = np.random.choice(21, p=probs) # aa from 0 to 20
        self.sequence[site] = new_aa

    def seq_energy(self):
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                sum+=self.J[self.sequence[i], self.sequence[j],i,j]
        return sum


#########
# this is done in the init function of the class
def initial_sample(L):
    list_nb=np.arange(21)
    init_sample=np.random.choice(list_nb,L)
    return init_sample


# calculate energy of a sequence given a coupling tensor - added to the class as "energy"
def plm_seq(seq,J):
    sum=0
    L=J.shape[-1]
    for i in range(L):
        for j in range(L):
            sum+=J[seq[i],seq[j],i,j]
    return sum

# energy diffs - 2 methods & functions to generate sequences
def plm_aa_calc_diff(aa_new,ind_change,seq,J_tens):
    L=J_tens.shape[-1]
    sum=0
    for j in range(L):
        delta_J1=J_tens[aa_new,seq[j],ind_change,j]-J_tens[seq[ind_change],seq[j],ind_change,j]
        delta_J2=J_tens[seq[j],aa_new,j,ind_change]-J_tens[seq[j],seq[ind_change],j,ind_change]
        sum+=delta_J1+delta_J2
    return sum

def plm_aa_calc_diff_alter(aa_new,ind_change,seq,J_tens):
    L=J_tens.shape[-1]
    sum=0
    for j in range(L):
        delta_J1=J_tens[aa_new,seq[j],ind_change,j]
        delta_J2=J_tens[seq[j],aa_new,j,ind_change]
        sum+=delta_J1+delta_J2
    return sum


def plm_ind_change(aa_pot,ind,seq,J,beta=1,old_plm=None,test=False):#probablement false
    if old_plm==None:
        old_plm=plm_seq(seq,J)
    list_aa_plm=np.array([])
    list_aa_plm_alter=np.array([])
    
    for i in range(len(aa_pot)):
        list_aa_plm=np.append(list_aa_plm,plm_aa_calc_diff(aa_pot[i],ind,seq,J))
        list_aa_plm_alter=np.append(list_aa_plm_alter,plm_aa_calc_diff_alter(aa_pot[i],ind,seq,J))
        # print("diff for two sequences",plm_aa_calc_diff(aa_pot[i],ind,seq,J))
        
        if test==True:
            seq_t=seq.copy()
            seq_t[ind]=aa_pot[i]
            print("diff two methods:",old_plm+plm_aa_calc_diff(aa_pot[i],ind,seq,J)-plm_seq(seq_t,J))
    prob_unn=np.exp(beta*list_aa_plm)    #à checker signe dans l'exponentiel
    pro_alter=np.exp(beta*list_aa_plm_alter)
    def assert_no_nan(arr, name="array"):
        """
        Checks if a NumPy array contains any NaNs.
        If it does, raises a ValueError with debug info.
        
        Parameters:
            arr (np.ndarray): Array to check.
            name (str): Optional name to include in the error message.
        """
        if np.isnan(arr).any():
            nan_indices = np.argwhere(np.isnan(arr))
            sample = arr.flatten()[np.isnan(arr.flatten())][:5]  # First 5 NaNs
            raise ValueError(
                f"NaN detected in {name}!\n"
                f"First few NaN values: {sample}\n"
                f"Indices of NaNs: {nan_indices[:5]}\n"
                f"Total NaNs: {np.isnan(arr).sum()}\n"
                f"old plm: {old_plm}\n"
                f"prob_list: {prob_unn}"
            )
    assert_no_nan(prob_unn/np.sum(prob_unn))
    print("difference of proba(should be around zero):",prob_unn/np.sum(prob_unn)-pro_alter/np.sum(pro_alter))
    return prob_unn/np.sum(prob_unn), list_aa_plm

def plm_ind_change_quick(aa_pot,ind,seq,J,beta=1):
    list_aa_plm=np.zeros(len(aa_pot))
    #list_aa_plm=np.array([])
    for i in range(len(aa_pot)):
        #list_aa_plm=np.append(list_aa_plm,plm_aa_calc_diff_alter(aa_pot[i],ind,seq,J))
        list_aa_plm[i]=plm_aa_calc_diff_alter(aa_pot[i],ind,seq,J)
    prob_unn=np.exp(beta*list_aa_plm) 
    return prob_unn/np.sum(prob_unn)

def plm_sample(aa_pot,proba_list):
    ind_choice=np.random.choice(range(len(aa_pot)),p=proba_list)
    return aa_pot[ind_choice],ind_choice


def generate_plm_y(J_tens,maxiter=10000,beta=1,initial_seq=None,quick=True):
    L=J_tens.shape[-1]
    if initial_seq is None:
        sequence=initial_sample(L)
    else:
        sequence=initial_seq
    list_nb=np.arange(21)
    seq_list=[]
    seq_list.append(sequence.copy())
    position_list=np.arange(L)
    ind_change_list=np.random.choice(position_list,size=maxiter)
    if quick:
        for i in tqdm(range(maxiter)):
            ind_change=ind_change_list[i]
            prob_list=plm_ind_change_quick(list_nb,ind_change,sequence,J_tens,beta=beta)
            new_aa,ind_chosen=plm_sample(list_nb,prob_list)
            sequence[ind_change]=new_aa
            seq_list.append(sequence.copy())
        return np.array(seq_list)
    else:
        old_plm=plm_seq(sequence,J_tens)
        plm_list=[]
        plm_list.append(old_plm)
        for i in tqdm(range(maxiter)):
            old_sequence=sequence.copy()
            ind_change=np.random.choice(np.arange(L))
            prob_list,pot_plm=plm_ind_change(list_nb,ind_change,sequence,J_tens,beta=beta,old_plm=old_plm)
            new_aa,ind_chosen=plm_sample(list_nb,prob_list)
            sequence[ind_change]=new_aa
            seq_list.append(sequence.copy())
            old_plm=pot_plm[ind_chosen]
            #print("difference in plm calc method:",(old_plm-plm_seq(sequence,J_tens))/old_plm) #seems close enough but not in the computer numerical errors range 
            plm_list.append(pot_plm[ind_chosen])
        return np.array(seq_list),plm_list