import numpy as np
from collections import Counter

# Define the 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def one_hot_encode(sequence):
    """Convert a protein sequence into a one-hot encoded matrix."""
    seq_length = len(sequence)
    encoding = np.zeros((seq_length, len(AMINO_ACIDS)), dtype=int)

    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            encoding[i, AMINO_ACIDS.index(aa)] = 1 # Set the position to 1

    return encoding

def amino_acid_composition(sequence):
    """Compute the frequency of each amino acid in a protein sequence."""
    total_length = len(sequence)
    aa_counts = Counter(sequence)

    # Normalize by total length
    composition = {aa: aa_counts.get(aa, 0) / total_length for aa in AMINO_ACIDS}
    return composition

# Example properties of amino acids
AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "molecular_weight": 89.1},
    "C": {"hydrophobicity": 2.5, "molecular_weight": 121.2},
    "D": {"hydrophobicity": -3.5, "molecular_weight": 133.1},
    "E": {"hydrophobicity": -3.5, "molecular_weight": 147.1},
    "F": {"hydrophobicity": 2.8, "molecular_weight": 165.2},
    "G": {"hydrophobicity": -0.4, "molecular_weight": 75.1},
    "H": {"hydrophobicity": -3.2, "molecular_weight": 155.2},
    "I": {"hydrophobicity": 4.5, "molecular_weight": 131.2},
    "K": {"hydrophobicity": -3.9, "molecular_weight": 146.2},
    "L": {"hydrophobicity": 3.8, "molecular_weight": 131.2},
    "M": {"hydrophobicity": 1.9, "molecular_weight": 149.2},
    "N": {"hydrophobicity": -3.5, "molecular_weight": 132.1},
    "P": {"hydrophobicity": -1.6, "molecular_weight": 115.1},
    "Q": {"hydrophobicity": -3.5, "molecular_weight": 146.2},
    "R": {"hydrophobicity": -4.5, "molecular_weight": 174.2},
    "S": {"hydrophobicity": -0.8, "molecular_weight": 105.1},
    "T": {"hydrophobicity": -0.7, "molecular_weight": 119.1},
    "V": {"hydrophobicity": 4.2, "molecular_weight": 117.1},
    "W": {"hydrophobicity": -0.9, "molecular_weight": 204.2},
    "Y": {"hydrophobicity": -1.3, "molecular_weight": 181.2},
}

def extract_properties(sequence):
    """Compute average physicochemical properties of a protein sequence."""
    properties = {"hydrophobicity": 0, "molecular_weight": 0}

    for aa in sequence:
        if aa in AA_PROPERTIES:
            properties["hydrophobicity"] += AA_PROPERTIES[aa]["hydrophobicity"]
            properties["molecular_weight"] += AA_PROPERTIES[aa]["molecular_weight"]

    # Normalize by sequence length
    properties = {key: value / len(sequence) for key, value in properties.items()}
    return properties

if __name__ == "__main__":
    sample_sequence = "MKTLLV"
    print("One-hot encoding:\n", one_hot_encode(sample_sequence))
    print("Amino acid composition:\n", amino_acid_composition(sample_sequence))
    print("Physicochemical properties:\n", extract_properties(sample_sequence))
