# PLM directory:

## Central classes and methods for plm generation
- plm_model.py:
    - SequencePLM class: initialize with J tensor, and optionally an initial sequence and beta value; contains methods to calculate PLM distribution and draw from it, calculate energy, convert sequence to letters
    - methods from Youssef: initial_sample() (implemented in class initialization), plm_seq() (implemented in class as seq_energy()), plm_aa_calc_diff(), plm_ind_change(), plm_sample(), generate_plm_y()
- plm_seq_utils.py: letter_to_num and num_to_letter dictionnaries, read_tensor_from_txt(), sequences_from_fasta(), modify_seq(), one_hot_seq_batch(), letters_to_nums() & nums_to_letters()/numbers_to_letters() - similar but requires additional input, seq_num_to_letters() 
- plm_gen_methods.py: generate_plm(), generate_plm_alter() - starts from new random sequence each time, generate_plm_n_save() - saves generated sequences as .npy and .txt

## Main for plm generation
- plm_gen_main.py: runs method to generate and save from provided initial sequence into "generated_sequences" in format: gen_seqs_w_init_seq_Ns{N_seqs}_r{ratio}; TODO: try different initial sequences to generate from, specify name when saving

## Files for analysis of generated sequences
- plm_PCA.py: provide generated sequences and test/train sequences, plot PCA and save in results/PCA_plots
- plm_hamming_dist.py: gen/train, test/train, gen/initial_seq, correlation plot
- plm_proba_distrib.py: sanity check to verify sampling of amino acids works as planned: plot proba distribution and empirical frequencies of draws, special check at HPD sites, saves in PLM/results/ProbaDsitrib

  ## Storing results
  - generated_sequences: contains plm generated sequences
  - results: contains results from analysis
