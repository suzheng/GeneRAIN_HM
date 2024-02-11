from Bio import SeqIO
from Bio import Align
from Bio.Seq import Seq
import time
import os
import statistics
import argparse
import pandas as pd
import json

from utils.utils import get_config
os.environ['SPECIES'] = "Mouse"
config = get_config()
from data_m.get_homolo_mg import get_homolo_mg
from dna.dna_utils import get_transcript2gene


# Function to perform pairwise alignment and return the normalized score
def pairwise_alignment(seq1, seq2, aligner):
    # Calculate the mean length of the two sequences
    mean_length = (len(seq1) + len(seq2)) / 2

    # Perform the alignment on the original sequences
    alignments = aligner.align(seq1, seq2)
    best_alignment = next(alignments)  # Assuming the first alignment is the best
    normal_score = best_alignment.score / mean_length  # Normalize by mean length
    
    # print("Alignment Score:", best_alignment.score)
    # print("Normalized Score:", normal_score)
    # print("Alignment:")
    # print(best_alignment) # Uses Biopython's format_alignment

    # Generate the reverse complement of seq2
    seq2_rc = Seq(seq2).reverse_complement()
    
    # Perform the alignment on the reverse complement
    alignments_rc = aligner.align(seq1, seq2_rc)
    best_alignment_rc = next(alignments_rc)  # Assuming the first alignment is the best
    reverse_complement_score = best_alignment_rc.score / mean_length  # Normalize by mean length
    
    # Return both normalized scores
    return max(normal_score, reverse_complement_score)


def get_seq_dict_from_fa_file(fa_file):
    def get_txn_from_fa_id(fa_id):
        return fa_id.split("_")[-1].split(".")[0]
    fasta_sequences = {get_txn_from_fa_id(record.id): record.seq for record in SeqIO.parse(fa_file, "fasta")}
    return fasta_sequences



def calculate_stats(float_list, key_prefix=""):
    # Calculate minimum and maximum
    min_score = min(float_list)
    max_score = max(float_list)

    # Calculate mean (average)
    mean_score = statistics.mean(float_list)

    # Calculate median
    median_score = statistics.median(float_list)

    list_len = len(float_list)
    # Calculate standard deviation
    std_dev = 0 if list_len == 1 else statistics.stdev(float_list)

    return {f"{key_prefix}n": list_len, 
            f"{key_prefix}min": min_score, 
            f"{key_prefix}max": max_score, 
            f"{key_prefix}mean": mean_score, 
            f"{key_prefix}median": median_score, 
            f"{key_prefix}std": std_dev}

def save_aligner_config(aligner, filename="aligner_config_backup.json"):
    # Prepare the configuration as a dictionary
    config = {
        "mode": aligner.mode,
        "match_score": aligner.match_score,
        "mismatch_score": aligner.mismatch_score,
        "open_gap_score": aligner.open_gap_score,
        "extend_gap_score": aligner.extend_gap_score
    }

    # Convert the configuration dictionary to a JSON string
    config_json = json.dumps(config, indent=4)

    # Write the JSON string to a file
    with open(filename, "w") as file:
        file.write(config_json)

    print(f"Aligner configuration has been saved to {filename}")

def calculate_gc_content(seq):
    # Convert the sequence to uppercase
    seq_upper = seq.upper()
    
    # Count G and C
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    
    # Calculate GC content
    gc_content = (gc_count / len(seq)) * 100
    return gc_content

def read_file_into_tuples(filename):
    tuples_list = []
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split()  # Adjust the split method if necessary
            if len(columns) == 2:  # Ensure there are exactly two columns
                tuples_list.append((columns[0], columns[1]))
    return tuples_list
    
def get_alignment_seq_stat_of_two_genes(human_gene, 
                                    mouse_gene,
                                    gene2transcripts,
                                    human_fa_dict,
                                    mouse_fa_dict,
                                    aligner,
                                    len_thres=100):
    human_txns = gene2transcripts.get(human_gene, None)
    if human_txns == None:
        return None
    valid_human_txns = [txn for txn in human_txns if txn in human_fa_dict and len(human_fa_dict[txn])>=len_thres]
    if len(valid_human_txns) == 0:
        return None
    mouse_txns = gene2transcripts.get(mouse_gene, None)
    if mouse_txns == None:
        return None
    valid_mouse_txns = [txn for txn in mouse_txns if txn in mouse_fa_dict and len(mouse_fa_dict[txn])>=len_thres]
    if len(valid_mouse_txns) == 0:
        return None
        
    best_scores_for_each_txn = []
    h_seq_len = []
    h_seq_gc = []
    for h_txn in valid_human_txns:
        scores = []
        human_txn_seq = human_fa_dict[h_txn]
        gc = calculate_gc_content(human_txn_seq)
        h_seq_gc.append(gc)
        h_seq_len.append(len(human_txn_seq))
        for m_txn in valid_mouse_txns:
            scores.append(pairwise_alignment(mouse_fa_dict[m_txn], human_txn_seq, aligner))
        best_scores_for_each_txn.append(max(scores))
    stat_of_best_scores_for_a_human_gene = calculate_stats(best_scores_for_each_txn, "h_align_")

    best_scores_for_each_txn = []
    m_seq_len = []
    m_seq_gc = []
    for m_txn in valid_mouse_txns:
        scores = []
        mouse_txn_seq = mouse_fa_dict[m_txn]
        gc = calculate_gc_content(mouse_txn_seq)
        m_seq_gc.append(gc)
        m_seq_len.append(len(mouse_txn_seq))
        for h_txn in valid_human_txns:
            scores.append(pairwise_alignment(mouse_txn_seq, human_fa_dict[h_txn], aligner))
        best_scores_for_each_txn.append(max(scores))
    stat_of_best_scores_for_a_mouse_gene = calculate_stats(best_scores_for_each_txn, "m_align_")
    h_seq_gc_mean = statistics.mean(h_seq_gc)
    m_seq_gc_mean = statistics.mean(m_seq_gc)
    h_seq_len_stat = calculate_stats(h_seq_len, "h_len_")
    m_seq_len_stat = calculate_stats(m_seq_len, "m_len_")
    ret_dict = {"h_gc_mean": h_seq_gc_mean,
                "m_gc_mean": m_seq_gc_mean,
                **stat_of_best_scores_for_a_human_gene,
                **stat_of_best_scores_for_a_mouse_gene,
                **h_seq_len_stat,
                **m_seq_len_stat
               }
    del ret_dict["h_len_n"]
    del ret_dict["m_len_n"]
    return ret_dict
def save_aligner_config(aligner, filename="aligner_config_backup.json"):
    # Prepare the configuration as a dictionary
    config = {
        "mode": aligner.mode,
        "match_score": aligner.match_score,
        "mismatch_score": aligner.mismatch_score,
        "open_gap_score": aligner.open_gap_score,
        "extend_gap_score": aligner.extend_gap_score
    }

    # Convert the configuration dictionary to a JSON string
    config_json = json.dumps(config, indent=4)

    # Write the JSON string to a file
    with open(filename, "w") as file:
        file.write(config_json)

    print(f"Aligner configuration has been saved to {filename}")

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Align sequences with specified parameters.')

    # Add arguments with default values
    parser.add_argument('--mode', type=str, default='global', help='Alignment mode (default: global)')
    parser.add_argument('--match_score', type=float, default=2, help='Match score (default: 2)')
    parser.add_argument('--mismatch_score', type=float, default=-1, help='Mismatch score (default: -1)')
    parser.add_argument('--open_gap_score', type=float, default=-2, help='Open gap score (default: -2)')
    parser.add_argument('--extend_gap_score', type=float, default=-1, help='Extend gap score (default: -1)')
    parser.add_argument('--seq_type', type=str, default='cds', help='Sequence type (default: cds), [cds, cds_utr, promoter_upstream_1kb, downstream_1kb, cds_utr_intron (large memory needed)]')
    parser.add_argument('--len_thres', type=int, default=100, help='Length threshold (default: 100)')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path, 2 columns, human genes on the left, non-human gene on the right, without gene prefix. h2m_genes for homologs')
    parser.add_argument('--output_file_prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--total_num_of_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--current_chunk_num', type=int, required=True, help='Current chunk number (1-based)')

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the aligner and set parameters
    aligner = Align.PairwiseAligner()
    aligner.mode = args.mode
    aligner.match_score = args.match_score
    aligner.mismatch_score = args.mismatch_score
    aligner.open_gap_score = args.open_gap_score
    aligner.extend_gap_score = args.extend_gap_score

    # Assign other variables
    seq_type = args.seq_type
    len_thres = args.len_thres
    input_file = args.input_file
    output_file_prefix = args.output_file_prefix
    total_num_of_chunks = args.total_num_of_chunks
    current_chunk_num = args.current_chunk_num

    dir_name = os.path.dirname(output_file_prefix)
    os.makedirs(dir_name, exist_ok=True)

    tsv_output = f"{output_file_prefix}.{seq_type}.{current_chunk_num}_of_{total_num_of_chunks}.tsv"
    config_output = f"{output_file_prefix}.{seq_type}.align_config.json"
    human_fa_file = f"{config.proj_path}/data/dna/seq_genes/human_GRCh38_GENCODEv44_{seq_type}.fa"
    mouse_fa_file = f"{config.proj_path}/data/dna/seq_genes/mouse_GRCm39_GENCODE_vm33_{seq_type}.fa"
    
    human_fa_dict = get_seq_dict_from_fa_file(human_fa_file)
    
    mouse_fa_dict = get_seq_dict_from_fa_file(mouse_fa_file)
    
    transcript2gene, gene2transcripts = get_transcript2gene()
    second_column_is_a_string_not_a_list = False
    if input_file == "h2m_genes":
        h2m_genes, m2h_genes, h2m_one2one, m2h_one2one = get_homolo_mg()
        gene_pair_list = list(h2m_genes.items())
    else:
        second_column_is_a_string_not_a_list = True
        gene_pair_list = read_file_into_tuples(input_file)
        
    stat_of_gene_pairs = []
    for i, (human_gene, mouse_genes) in enumerate(gene_pair_list):
        if i % total_num_of_chunks != current_chunk_num - 1:
            continue
        if second_column_is_a_string_not_a_list:
            mouse_genes = [mouse_genes]
        for mouse_gene in mouse_genes:
            mouse_gene = f"{config.gene_prefix}" + mouse_gene
            gene_pair = f"{human_gene} {mouse_gene}"
            print(gene_pair)
            stat_dict = get_alignment_seq_stat_of_two_genes(human_gene, 
                                                    mouse_gene,
                                                    gene2transcripts,
                                                    human_fa_dict,
                                                    mouse_fa_dict,
                                                    aligner,
                                                    len_thres=100)
            if stat_dict != None:
                ret_dict={"gene_pair": gene_pair, **stat_dict}
                stat_of_gene_pairs.append(ret_dict)
    
    
    df = pd.DataFrame(stat_of_gene_pairs)
    
    # Write the DataFrame to a TSV file
    df.to_csv(tsv_output, sep='\t', index=False)
    print(f"Saved to {tsv_output} .")
    save_aligner_config(aligner, filename=config_output)
if __name__ == '__main__':
    main() 