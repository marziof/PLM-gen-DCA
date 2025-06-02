# PLM directory:

## Central classes and methods generation
- plm_model.py:
    - SequencePLM class: initialize with J tensor, and optionally an initial sequence and beta value; contains methods to calculate PLM distribution and draw from it, calculate energy, convert sequence to letters
    - plm_gen_methods.py: generate_plm(), generate_plm_alter() - starts from new random sequence each time, generate_plm_n_save() - saves generated sequences as .npy and .txt
    - plm_gen_main.py: runs method to generate and save from provided initial sequence into "generated_sequences" in format: gen_seqs_w_init_seq_Ns{N_seqs}_r{ratio}
- gillespie.py:
    - SequenceGill class: initialize the same as for PLM. Draw an amino acid using .draw_aa().
    - gillespie_main.py: implements a function generate_gill_n_save() to do multiple iterations and save them in a directory.
    - gillespie_main_nb.ipynb: Runs the gillespie sampling algorithm for different conditions (initial sequence, sampling temperature...). An initial study of the generated sequences is also done in this notebook.
- monte_carlo.py:
    - SequenceMC: Same initialization as previous classes. To draw an amino acid according to MCMC use the method draw_aa_metropolis().
    - montecarlo_main.py: implements a function generate_mc_n_save() to do multiple iterations and save them in a directory.
- temperatures.ipynb: Runs MC and PLM sampling for different conditions (initial sequence, sampling temperature...). An initial study of the generated sequences is also done in this notebook. 
- seq_utils.py: letter_to_num and num_to_letter dictionnaries, read_tensor_from_txt(), sequences_from_fasta(), modify_seq(), one_hot_seq_batch(), letters_to_nums() & nums_to_letters()/numbers_to_letters() - similar but requires additional input, seq_num_to_letters() 
  


## Files for analysis of generated sequences
- PCA_func.py: provides the functions to do a PCA analysis plots.
- hamming_dist.py: Provides methods to calculate the hamming distance between sequences. Used for decorrelation graphs.
- check_proba_distrib.py (in extra): sanity check to verify sampling of amino acids works as planned: plot proba distribution and empirical frequencies of draws, special check at HPD sites, saves in PLM/results/ProbaDsitrib
- decorrelation_and_frequency_plots.ipynb: Used for the study of the decorrelation plots for the different sampling methods and for the aa frquency graphs as well.
- ESM_pymol_run.ipynb: Compares the RMSD (3D structure) between the different sequence groups (generated vs true sequences)
- Blast_nb.ipynb and Read_Blast.py: analyze blast results
- phase_transition.ipynb: Studies the evolution of different quantities (hamming distance, KL divergence over PCA distribution and PCA closest point distance) over the change of the temperature.

  ## Storing results
  - generated_sequences: contains plm generated sequences
  - gill_generated_sequences:contains gillespie generated sequences
  - mc_genereated sequences:contains MCMC generated sequences
  - results: contains results from analysis
