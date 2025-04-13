import numpy as np

letter_to_num = {
            'A': 1,  'B': 21, 'C': 2,  'D': 3,  'E': 4,
            'F': 5,  'G': 6,  'H': 7,  'I': 8,  'J': 21,
            'K': 9,  'L': 10, 'M': 11, 'N': 12, 'O': 21,
            'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
            'U': 21, 'V': 18, 'W': 19, 'X': 21, 'Y': 20,
            '-': 21  # Gap symbol
        }

num_to_letter  = {v: k for k, v in letter_to_num.items()}

def sequences_from_fasta(file_path):
    "Reads fasta file and returns list of letter sequences"
    sequences = []
    with open(file_path, "r") as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Ignore headers
                if sequence:  # If there's a previous sequence, add it to the list
                    sequences.append(sequence)
                sequence = ''  # Reset sequence for the next one
            else:
                sequence += line  # Add the line to the current sequence
        if sequence:  # Append the last sequence
            sequences.append(sequence)
    return sequences

def letters_to_nums(sequence):
    "receives AA sequence (format ABC... and reutrns [1 2 3 ...])"
    return np.array([letter_to_num.get(aa, 21) for aa in sequence])

def nums_to_letters(sequence):
    """Receives a sequence of integers (e.g., [1, 2, 4]) and returns a string of corresponding amino acid letters."""
    num_to_letter = {v: k for k, v in letter_to_num.items()}
    return ''.join([num_to_letter.get(num, 'X') for num in sequence])


def nums_to_letters1(sequence):
    """
    Receives a sequence of integers (e.g., [1, 2, 4])
    and returns a string of corresponding amino acid letters.
    Also prints the conversion steps.
    """
    # Reverse mapping
    num_to_letter = {v: k for k, v in letter_to_num.items()}
    
    print("num_to_letter dictionary:")
    for num, letter in sorted(num_to_letter.items()):
        print(f"{num}: '{letter}'")
    
    # Step-by-step conversion
    result = ''
    print("\nConversion steps:")
    for num in sequence:
        letter = num_to_letter.get(num, 'X')
        print(f"{num} → '{letter}'")
        result += letter
    
    print("\nFinal result:", result)
    return result
