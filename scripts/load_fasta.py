from Bio import SeqIO

def load_fasta_sequences(fasta_path):
    Sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        Sequences[record.id] = str(record.seq)
    return Sequences

if __name__ == "__main__":
    sample_sequences = load_fasta_sequences("data/sample.fasta")
    print(sample_sequences)