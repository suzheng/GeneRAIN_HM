import os
from eval.visual_utils import filter_dataframe
import numpy as np
from eval_m.m_eval_utils import read_tsv_to_dict
def write_string_to_file(file_path, string_data):
    # Get the directory name from the file path
    dir_name = os.path.dirname(file_path)

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Write the string data to the file
    with open(file_path, 'w') as file:
        file.write(string_data + "\n")
    return f"String written to {file_path}"


import pandas as pd
import gzip

def read_gz_tsv(filename, comment_char='#', header='infer', parse_commented_header=False, parsed_header_index_offset=2):
    # Initialize an empty list to hold the header if parsing from comments
    parsed_header = None

    # Check if we need to parse the header from the commented lines
    if parse_commented_header:
        # Determine if the file is gzip-compressed and open accordingly
        open_func = gzip.open if filename.endswith('.gz') else open
        with open_func(filename, 'rt') as f:
            for line in f:
                if line.startswith(comment_char):
                    # Try to parse the header line
                    line = line.strip(comment_char + ' \n')
                    if ':' in line:  # Look for the separator between header name and description
                        header_name = line.split(':')[0].strip()
                        if parsed_header is None:
                            parsed_header = []
                        parsed_header.append(header_name)
                else:
                    break  # Stop reading after comments

    # Define read options based on the parameters
    read_opts = {
        'sep': '\t',
        'comment': comment_char
    }

    # Use the parsed header if one was found, else use the specified header option
    if parsed_header:
        read_opts['names'] = parsed_header[parsed_header_index_offset:]
        read_opts['header'] = None  # Ensure the actual header line isn't treated as data
    else:
        read_opts['header'] = header

    # Check if file is in gzip format
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:  # open the gzip file
            df = pd.read_csv(f, **read_opts)  # use Pandas to read the TSV with the options
    else:
        df = pd.read_csv(filename, **read_opts)  # directly read the file with Pandas with the options
    
    return df

def print_unique_values(df):
    for column in df.columns:
        # Get unique values and counts for the column
        unique_counts = df[column].value_counts()
        
        # Check if unique values are less than 10
        if len(unique_counts) < 10:
            # Convert the counts to a sorted dictionary
            sorted_counts_dict = unique_counts.sort_values(ascending=False).to_dict()
            
            # Format and print the dictionary as a list of strings suitable for Python
            formatted_list = [f"'{k}':{v}\n" for k, v in sorted_counts_dict.items()]
            print(f"Column: '{column}' has less than 10 unique values: [{', '.join(formatted_list)}]")
            formatted_list = [f"'{k}'" for k, v in sorted_counts_dict.items()]
            print(f"Column: '{column}' has less than 10 unique values: [{','.join(formatted_list)}]")
            
            print("\n")  # Add a newline for better readability between columns

def df_cols_to_dict(df, key_col, value_col):
    # Initialize an empty dictionary to hold the result
    result_dict = {}

    # Ensure the columns exist in the DataFrame
    if key_col in df.columns and value_col in df.columns:
        # Loop through each row in the DataFrame
        for _, row in df.iterrows():
            key = row[key_col]
            value = row[value_col]

            # If the key isn't already in the result dict, add it with an empty list
            if key not in result_dict:
                result_dict[key] = []
            if value not in result_dict[key]:
            # Append the current value to the list of values for this key
                result_dict[key].append(value)
    else:
        raise ValueError("One or both specified columns do not exist in the DataFrame")

    return result_dict


def query_genes(human_gene, mouse_gene, human_dict, mouse_dict, return_intersect_ratio=False):
    """
    human_dict: {human_gene: [item1, item2, item3,..], ..}
    mouse_dict: {mouse_gene: [item1, item2, item3,..], ..}
    """
    return_value = None
    intersection_len = 0
    intersection_list = []
    # Check if the human gene is in human_dict and mouse gene is not in mouse_dict
    if human_gene in human_dict and mouse_gene not in mouse_dict:
        return_value = "HumanNotMouse"
    
    # Check if the mouse gene is in mouse_dict and human gene is not in human_dict
    elif mouse_gene in mouse_dict and human_gene not in human_dict:
        return_value = "MouseNotHuman"
    
    # Check if neither gene is found in either dictionary
    elif human_gene not in human_dict and mouse_gene not in mouse_dict:
        return_value = "NeitherFound"
    
    # Check if both are found but values have no intersection
    elif human_gene in human_dict and mouse_gene in mouse_dict:
        # Extract the associated values and convert to sets for intersection operation
        human_values = set(human_dict[human_gene])
        mouse_values = set(mouse_dict[mouse_gene])
        
        # Calculate the intersection and union
        intersection = human_values.intersection(mouse_values)
        union = human_values.union(mouse_values)
        
        # Check for intersection
        if not intersection:
            return_value = "BothNoIntersect"
        else:
            intersection_len = len(intersection)
            intersection_list = list(intersection)
            # Calculate the intersection ratio
            intersection_ratio = len(intersection) / (len(union)/2)
            if return_intersect_ratio:
                return_value = f"BothIntersect:{intersection_ratio:.2f}"
            else:
                return_value = f"BothIntersect"

    # Return the result
    return return_value, intersection_len, intersection_list

class DiseaseAllianceReader:
    def __init__(self, config):
        self.config = config
        h_input_file = f"{config.proj_path}/data/external/alliancegenome/DISEASE-ALLIANCE_HUMAN.tsv.gz"
        h_df = read_gz_tsv(h_input_file)
        h_df_filt0 = filter_dataframe(h_df, {'AssociationType':['is_implicated_in','is_marker_for'],
                                         })

        h_df_filt1 = filter_dataframe(h_df, {'AssociationType':['is_implicated_in','is_marker_for'],
                                          'EvidenceCodeName': ['inference by association of genotype from phenotype used in manual assertion','expression pattern evidence used in manual assertion','direct assay evidence used in manual assertion','mutant phenotype evidence used in manual assertion','genetic interaction evidence used in manual assertion']
                                         })
        h_df_filt2 = filter_dataframe(h_df, {'AssociationType':['is_implicated_in','is_marker_for'],
                                          'EvidenceCodeName': ['inference by association of genotype from phenotype used in manual assertion','direct assay evidence used in manual assertion','mutant phenotype evidence used in manual assertion']
                                         })
        
        self.h_gene2att0 = df_cols_to_dict(h_df_filt0, "DBObjectSymbol", "DOtermName")
        self.h_gene2att1 = df_cols_to_dict(h_df_filt1, "DBObjectSymbol", "DOtermName")
        self.h_gene2att2 = df_cols_to_dict(h_df_filt2, "DBObjectSymbol", "DOtermName")
        
        m_input_file = f"{config.proj_path}/data/external/alliancegenome/DISEASE-ALLIANCE_MGI.tsv.gz"
        m_df = read_gz_tsv(m_input_file)
        
        m_df_filt0 = filter_dataframe(m_df, {'DBobjectType':['gene'],
                                            'AssociationType':['is_implicated_in','is_marker_for']
                                            })
        
        m_df_filt1 = filter_dataframe(m_df_filt0, {'DBobjectType':['gene'],
                                                   'AssociationType':['is_implicated_in','is_marker_for'],
                                                    'EvidenceCodeName': ['evidence used in automatic assertion','author statement supported by traceable reference']
                                                  })
        
        m_df_filt2 = filter_dataframe(m_df_filt0, {'DBobjectType':['gene'],
                                                   'AssociationType':['is_implicated_in','is_marker_for'],
                                                    'EvidenceCodeName': ['author statement supported by traceable reference']
                                                  })

        self.m_gene2att0 = df_cols_to_dict(m_df_filt0, "DBObjectSymbol", "DOtermName")
        self.m_gene2att1 = df_cols_to_dict(m_df_filt1, "DBObjectSymbol", "DOtermName")
        self.m_gene2att2 = df_cols_to_dict(m_df_filt2, "DBObjectSymbol", "DOtermName")

    def query_genes(self, human_gene, mouse_gene, return_intersect_ratio=False):
        if self.config.gene_prefix in mouse_gene:
            mouse_gene = mouse_gene.replace(self.config.gene_prefix, "")
        res0, shared_num0, shared_list0 = query_genes(human_gene, mouse_gene, self.h_gene2att0, self.m_gene2att0, return_intersect_ratio=return_intersect_ratio)
        res1, shared_num1, shared_list1 = query_genes(human_gene, mouse_gene, self.h_gene2att1, self.m_gene2att1, return_intersect_ratio=return_intersect_ratio)
        res2, shared_num2, shared_list2 = query_genes(human_gene, mouse_gene, self.h_gene2att2, self.m_gene2att2, return_intersect_ratio=return_intersect_ratio)
        
        ret_dict = {"alliance_disease_full": res0,
                    "alliance_disease_filt1": res1,
                    "alliance_disease_filt2": res2,
                    "alliance_disease_shared_num_full": shared_num0,
                    "alliance_disease_shared_num_filt1": shared_num1,
                    "alliance_disease_shared_num_filt2": shared_num2,
                    "alliance_disease_shared_list_full": shared_list0,
                    "alliance_disease_shared_list_filt1": shared_list1,
                    "alliance_disease_shared_list_filt2": shared_list2,
                   }
        return ret_dict

def read_HMD_HumanPhenotype(config):
    # Initialize an empty dictionary
    file_path = f'{config.proj_path}/data/external/Homolo/MouseGenomeInfo/HMD_HumanPhenotype.rpt'
    gene_pair2mp_list = {}
    gene_pair2len = {}
    # Open and read the TSV file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split('\t')
            
            key = columns[0] + f" {config.gene_prefix}" + columns[2]            
            # Check if the line has the required number of columns
            if len(columns) >= 5:
                # Split the 5th column by ", " to form the list of values
                values = columns[4].split(', ')
            else:
                values = []
            gene_pair2mp_list[key] = values
            gene_pair2len[key] = len(values)
    return gene_pair2mp_list, gene_pair2len


def read_MGI_DiseaseGeneModel(config):
    """
    return MGI_DiseaseGeneModel2diseases, MGI_DiseaseGeneModel2len, MGI_DiseaseGeneModel_with_m_geno_2diseases, MGI_DiseaseGeneModel_with_m_geno_2len
    The *2diseases dicts are gene pairs to disease list dicts
    The *len dicts are gene pairs to disease list len
    *with_m_geno* are for results filtered for records with Mouse genotype
    """
    MGI_DiseaseGeneModel_df = read_gz_tsv(f"{config.proj_path}/data/external/Homolo/MouseGenomeInfo/MGI_DiseaseGeneModel.rpt",
                parse_commented_header=True
               )
    # Filter for record that with Mouse genotype
    MGI_DiseaseGeneModel_with_mouse_genotype_df = MGI_DiseaseGeneModel_df[MGI_DiseaseGeneModel_df["Mouse genotype IDs associated with DO term"].notna()]
    
    def MGI_DiseaseGeneModel_df_to_dict(MGI_DiseaseGeneModel_df):
        # Initialize an empty dictionary
        MGI_DiseaseGeneModel_dict = {}
        
        # Iterate over each row in the DataFrame
        for index, row in MGI_DiseaseGeneModel_df.iterrows():
            # Concatenate "Human Gene Symbol" and "Mouse Gene" to form the key
            key = str(row["Human Gene Symbol"]) + f" {config.gene_prefix}" + str(row["Mouse Gene"])
            
            # Append "DO term name associated with human gene" to the list in the dictionary
            if key not in MGI_DiseaseGeneModel_dict:
                MGI_DiseaseGeneModel_dict[key] = []
            MGI_DiseaseGeneModel_dict[key].append(row["DO term name associated with human gene"])
        gene_pair2len = {k: len(v) for k,v in MGI_DiseaseGeneModel_dict.items()}
        return MGI_DiseaseGeneModel_dict, gene_pair2len
    
    MGI_DiseaseGeneModel2diseases, MGI_DiseaseGeneModel2len = MGI_DiseaseGeneModel_df_to_dict(MGI_DiseaseGeneModel_df)
    MGI_DiseaseGeneModel_with_m_geno_2diseases, MGI_DiseaseGeneModel_with_m_geno_2len = MGI_DiseaseGeneModel_df_to_dict(MGI_DiseaseGeneModel_with_mouse_genotype_df)
    return MGI_DiseaseGeneModel2diseases, MGI_DiseaseGeneModel2len, MGI_DiseaseGeneModel_with_m_geno_2diseases, MGI_DiseaseGeneModel_with_m_geno_2len


from collections import defaultdict

class MGI_DO_Reader:
    def __init__(self, config):
        self.config = config
        mgi_do_df = read_gz_tsv(f"{config.proj_path}//data/external/Homolo/MouseGenomeInfo/MGI_DO.rpt")
        # Initialize the dictionaries with list as the default factory function
        self.human_dict = defaultdict(list)
        self.mouse_dict = defaultdict(list)
        for index, row in mgi_do_df.iterrows():
            if row['NCBI Taxon ID'] == 9606:
                self.human_dict[row['Symbol']].append(row['DO Disease Name'])
            elif row['NCBI Taxon ID'] == 10090:
                self.mouse_dict[row['Symbol']].append(row['DO Disease Name'])

    def query_genes(self, human_gene, mouse_gene, return_intersect_ratio=False):
        if self.config.gene_prefix in mouse_gene:
            mouse_gene = mouse_gene.replace(self.config.gene_prefix, "")
        res0, shared_num0, shared_list0 = query_genes(human_gene, mouse_gene, self.human_dict, self.mouse_dict, return_intersect_ratio=return_intersect_ratio)
        
        ret_dict = {"MGI_DO": res0,
                    "MGI_shared_num": shared_num0,
                   }
        return ret_dict

def split_human_mouse_emb_and_genes(gene_emb_npy, gene_names, m_config, to_lower_case=True):
    """
    return gene_emb_npy_human, gene_emb_npy_mouse, genes_human, genes_mouse
    """
    assert gene_emb_npy.shape[0] == len(gene_names)
    gene_types = np.array(["M" if m_config.gene_prefix in gene else "H" for gene in gene_names])
    gene_emb_npy_human = gene_emb_npy[gene_types=="H"]
    gene_emb_npy_mouse = gene_emb_npy[gene_types=="M"]
    
    print(f"gene_emb_npy_human.shape {gene_emb_npy_human.shape}")
    
    print(f"gene_emb_npy_mouse.shape {gene_emb_npy_mouse.shape}")
    
    genes_human = np.array(gene_names)[gene_types=="H"]
    genes_mouse = np.array(gene_names)[gene_types=="M"]
    if to_lower_case:
        genes_human = [gene.lower() for gene in genes_human]
        genes_mouse = [gene.lower() for gene in genes_mouse]

    return gene_emb_npy_human, gene_emb_npy_mouse, genes_human, genes_mouse

import math

def split_dict(input_dict, split_ratio=0.95):
    # Determine the size of the dictionary and the split count
    total_size = len(input_dict)
    split_count = math.ceil(total_size * split_ratio)

    # Create two new dictionaries for the split
    part1 = dict(list(input_dict.items())[:split_count])
    part2 = dict(list(input_dict.items())[split_count:])

    return part1, part2

def dict_to_file(dict_data, file_name, m_config):
    """
    Writes a dictionary to a text file, with each key-value pair on a new line,
    and the key and value separated by a space or tab.

    Parameters:
    dict_data (dict): The dictionary to write to the file.
    file_name (str): The name of the file to which to write the dictionary.
    """
    with open(file_name, 'w') as file:
        for key, value in dict_data.items():
            file.write(f"{m_config.gene_prefix}{key.lower()}\t{value.lower()}\n")
    print(f"Saved to {file_name}")

def save_matrix_and_ids(matrix, ids, output_file, ids_file_path=None):
    assert matrix.shape[0] == len(ids)
    output_dir = os.path.dirname(output_file)

    # Create the directory if it does not exist
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    
    # Save the matrix to a file in binary format
    np.save(output_file, matrix)  # This will save in .npy format
    
    # Determine the IDs file path if not provided
    if ids_file_path is None:
        ids_file_path = f"{output_file}_ids.txt"
    
    # Save the list of IDs to a file
    with open(ids_file_path, 'w') as f:
        for id in ids:
            f.write(f"{id}\n")

def load_matrix_and_ids(matrix_file, ids_file_path=None):
    # Load the matrix from the file
    matrix = np.load(matrix_file)  # Assuming the file is in .npy format
    if ids_file_path == None:
        ids_file_path = f"{matrix_file}_ids.txt"
    # Load the list of IDs from the file
    ids = []
    with open(ids_file_path, 'r') as f:
        ids = f.read().splitlines()

    return matrix, ids
import glob
from data_m.get_homolo_mg import get_homolo_mg
def get_combined_dna_seq_conserv_df(seq_type, config):
    def read_dna_seq_of_gene_pairs_conserv_results(seq_type, gene_list_name, total_num_of_chunks, config):
        output_file_prefix = f"{config.proj_path}/results/anal/dna/dna_seq_of_gene_pairs/{gene_list_name}"
        tsv_files = glob.glob(f"{output_file_prefix}.{seq_type}.*_of_{total_num_of_chunks}.tsv")
        if len(tsv_files) != total_num_of_chunks:
            print(f"Only {len(tsv_files)} of {total_num_of_chunks} files found!")
        
        dataframes = [pd.read_csv(file, sep='\t') for file in tsv_files]
        combined_df = pd.concat(dataframes)
        return combined_df
    h2m_genes, m2h_genes, h2m_one2one, m2h_one2one = get_homolo_mg()
    all_homolog_gene_pairs_human_on_left = [f"{k} {config.gene_prefix}{gene}" for k, v in h2m_genes.items() for gene in v]
    
    
    # seq_type = "promoter_upstream_1kb" # cds, cds_utr, promoter_upstream_1kb, downstream_1kb
    
    gene_list_name = "output_h2m_genes"
    total_num_of_chunks = 50
    dna_cons_df = read_dna_seq_of_gene_pairs_conserv_results(seq_type, gene_list_name, total_num_of_chunks, config)
    
    gene_list_name = "exp16_hm_BERT_coding.epoch20.full_spv_1st"
    total_num_of_chunks = 100
    conserv_for_non_homolog_closest_emb_gene_pair_df = read_dna_seq_of_gene_pairs_conserv_results(seq_type, gene_list_name, total_num_of_chunks, config)
    
    gene_list_name = "random_gene_pairs_exp16_seed8"
    total_num_of_chunks = 100
    random_gene_pairs_cons_df = read_dna_seq_of_gene_pairs_conserv_results(seq_type, gene_list_name, total_num_of_chunks, config)                                                                                            
    
    gene_list_name = "ave_supAlign_and_sharedEmbSupAlign"
    total_num_of_chunks = 100
    dna_ave_df = read_dna_seq_of_gene_pairs_conserv_results(seq_type, gene_list_name, total_num_of_chunks, config)  

    # dna_cons_df["gene_pair_group"] = "one2one_homolog"
    # conserv_for_non_homolog_closest_emb_gene_pair_df["gene_pair_group"] = "non_homolog_closest_emb"
    # random_gene_pairs_cons_df["gene_pair_group"] = "random_gene_pairs"
    
    conserv_for_non_homolog_closest_emb_gene_pair_df = conserv_for_non_homolog_closest_emb_gene_pair_df[
        ~conserv_for_non_homolog_closest_emb_gene_pair_df['gene_pair'].isin(all_homolog_gene_pairs_human_on_left)
    ]
    dna_cons_df_combined = pd.concat([dna_cons_df, conserv_for_non_homolog_closest_emb_gene_pair_df, random_gene_pairs_cons_df, dna_ave_df])
    # dna_cons_df_combined[['h_gene', 'm_gene']] = dna_cons_df_combined['gene_pair'].str.split(' ', expand=True)
    return dna_cons_df_combined.drop_duplicates()
    
def add_anno_info(dna_cons_df_combined, config):
    """
    dna_cons_df_combined: column gene_pair is needed. but not h_gene, m_gene
    """
    if 'h_gene' not in dna_cons_df_combined.columns:
        dna_cons_df_combined[['h_gene', 'm_gene']] = dna_cons_df_combined['gene_pair'].str.split(' ', expand=True)
    ## Read DiseaseAlliance
    gpar = DiseaseAllianceReader(config)
    disease_alliance_expanded_df = dna_cons_df_combined.apply(lambda row: pd.Series(gpar.query_genes(row['h_gene'], row['m_gene'])), axis=1)

    ## read HMD_HumanPhenotype
    gene_pair2mp_list, gene_pair2len = read_HMD_HumanPhenotype(config)
    gene_pair2bool = {k: True if v>0 else False for k,v in gene_pair2len.items()}
    
    dna_cons_df_combined['HMD_HumanPhenotype_num'] = dna_cons_df_combined.apply(lambda row: gene_pair2len.get(row['gene_pair'], 0), axis=1)
    dna_cons_df_combined['HMD_HumanPhenotype_bool'] = dna_cons_df_combined.apply(lambda row: gene_pair2bool.get(row['gene_pair'], False), axis=1)
    dna_cons_df_combined['HMD_HumanPhenotype_phenos'] = dna_cons_df_combined.apply(lambda row: gene_pair2mp_list.get(row['gene_pair'], []), axis=1)
    
    ## read MGI_DiseaseGeneModel
    MGI_DiseaseGeneModel2diseases, MGI_DiseaseGeneModel2len, MGI_DiseaseGeneModel_with_m_geno_2diseases, MGI_DiseaseGeneModel_with_m_geno_2len = read_MGI_DiseaseGeneModel(config)
    
    dna_cons_df_combined['MGI_DiseaseGeneModel'] = dna_cons_df_combined.apply(lambda row: MGI_DiseaseGeneModel2len.get(row['gene_pair'], 0), axis=1)
    dna_cons_df_combined['MGI_DiseaseGeneModel_diseases'] = dna_cons_df_combined.apply(lambda row: MGI_DiseaseGeneModel2diseases.get(row['gene_pair'], []), axis=1)
    dna_cons_df_combined['MGI_DiseaseGeneModel_with_m_geno'] = dna_cons_df_combined.apply(lambda row: MGI_DiseaseGeneModel_with_m_geno_2len.get(row['gene_pair'], 0), axis=1)

    ## read MGI_DO
    mgi_do = MGI_DO_Reader(config)
    mgi_do_expanded_df = dna_cons_df_combined.apply(lambda row: pd.Series(mgi_do.query_genes(row['h_gene'], row['m_gene'])), axis=1)
    
    dna_cons_df_combined_expanded = pd.concat([dna_cons_df_combined, disease_alliance_expanded_df, mgi_do_expanded_df], axis=1)
    return dna_cons_df_combined_expanded

import pandas as pd

def list_of_dicts_to_dataframe(dicts, labels):
    # Check if the length of dicts and labels are the same
    if len(dicts) != len(labels):
        raise ValueError("The number of dictionaries and labels must be the same.")

    # Find the shared keys in all dictionaries
    shared_keys = set.intersection(*(set(d.keys()) for d in dicts))

    # Convert shared_keys to a list to ensure order
    shared_keys_list = list(shared_keys)

    # Extract the values for the shared keys
    data = {label: [d.get(key) for key in shared_keys_list] for d, label in zip(dicts, labels)}

    # Create and return the DataFrame
    return pd.DataFrame(data, index=shared_keys_list)

def get_gene2type_dict_from_meta(config):
    mouse_gene2type = read_tsv_to_dict(f"{config.proj_path}//data/external/ARCHS/mouse_gene_v2.2.h5.genes_meta.tsv",
                    key_col_idx=3, 
                    value_col_idx=0
                    )
    mouse_gene2type2 = {f"{config.gene_prefix}{k}": v for k, v in mouse_gene2type.items()}
    # genes with identical names have same type
    human_gene2type = read_tsv_to_dict(f"{config.proj_path}/../DeepGeneNet/data/external/ARCHS/human_gene_v2.2.h5.genes_meta.tsv",
                                       key_col_idx=2, 
                                       value_col_idx=0
                                      )
    gene2type_dict ={**mouse_gene2type, **mouse_gene2type2, **human_gene2type}
    return gene2type_dict
    


def get_gene2type_dict_from_Genecode(config):
    mouse_gene2type = read_tsv_to_dict(f"{config.proj_path}/data/external/Gencode/gencode.vM33.annotation.gff3.gz.gene_type.tsv",
                    key_col_idx=0, 
                    value_col_idx=1
                    )
    mouse_gene2type2 = {f"{config.gene_prefix}{k}": v for k, v in mouse_gene2type.items()}
    # genes with identical names have same type
    human_gene2type = read_tsv_to_dict(f"{config.proj_path}/data/external/Gencode/gencode.v44.annotation.gff3.gz.gene_type.tsv",
                                       key_col_idx=0, 
                                       value_col_idx=1
                                      )
    gene2type_dict ={**mouse_gene2type, **mouse_gene2type2, **human_gene2type}
    return gene2type_dict

def get_gene2type_dict(config):
    gene2type_dict_from_meta = get_gene2type_dict_from_meta(config)
    gene2type_dict_from_Genecode = get_gene2type_dict_from_Genecode(config)
    missed_by_genecode = {k: v for k, v in gene2type_dict_from_meta.items() if k not in gene2type_dict_from_Genecode}
    gene2type_dict = {**missed_by_genecode, **gene2type_dict_from_Genecode}
    return gene2type_dict