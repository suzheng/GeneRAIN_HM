import numpy as np
import os
import glob
import os

os.environ['SPECIES'] = "Mouse"
from eval.visual_utils import Naming_Json
njson = Naming_Json()

def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        # Skip the first line as it contains the dimensions information
        next(file)
        tokens = []
        embeddings = []
        for line in file:
            parts = line.strip().split()
            # The first part is the token name, the rest are the embedding values
            tokens.append(parts[0])
            embeddings.append([float(val) for val in parts[1:]])
    print(f"Loaded emb from file_path!")
    return tokens, embeddings

def load_muse_result(muse_result_dir, gene_to_ori_case_dict=None):
    """
    return hm_emb, hm_gene_names, hm_token_types
    """
    muse_output_mouse = f"{muse_result_dir}/vectors-mouse.txt"
    mouse_genes, mouse_emb = load_embeddings(muse_output_mouse)
    
    muse_output_human = f"{muse_result_dir}/vectors-human.txt"
    human_genes, human_emb = load_embeddings(muse_output_human)
    
    hm_gene_names = human_genes + mouse_genes
    if gene_to_ori_case_dict != None:
        hm_gene_names = [gene_to_ori_case_dict[gene] for gene in hm_gene_names]
    hm_token_types = ([njson.h_label] * len(human_genes)) + ([njson.m_label]*len(mouse_genes)) 
    
    hm_emb = np.vstack((human_emb, mouse_emb))
    return hm_emb, hm_gene_names, hm_token_types

def get_gene_to_ori_case_dict(gene2idx):

    gene_to_ori_case_dict = {gene.lower(): gene  for gene in gene2idx.keys()}
    gene_to_ori_case_dict.update({gene: gene  for gene in gene2idx.keys()})
    return gene_to_ori_case_dict

def get_muse_result_uid_dir(dir_without_uid):
    """
    input dir_without_uid, return dir_with_uid
    """
    muse_result_files = glob.glob(f"{dir_without_uid}/*/vectors-human.txt")
    muse_result_dir = None
    if len(muse_result_files) == 1:
        muse_result_dir = os.path.dirname(muse_result_files[0])
    else:
        print("more than one file found for muse_result_files")
    return muse_result_dir