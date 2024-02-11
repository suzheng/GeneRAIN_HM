from utils.params import params
params = params()
import os
import h5py
import torch
import numpy as np
import pandas as pd
import sys
from data_m.MixedDataset import MixedDataset

# param_json_file = "/g/data/yr31/zs2131/tasks/2023/RNA_expr_net/anal_mouse/training/exp6/exp6_hm_GPT_coding.param_config.json"
# os.environ['PARAM_JSON_FILE'] = param_json_file
def get_non_dup_and_qualified_mean_bool_vectors(gene_stat_uniq_bool_file, min_mean_val_for_zscore):
    # Read the TSV file
    df = pd.read_csv(gene_stat_uniq_bool_file, sep='\t')

    # Create a numpy boolean vector from the 'max_mean_nondup' column
    max_mean_nondup_vector = df['max_mean_nondup'].values.astype(bool)

    # Create a numpy boolean vector that indicates whether the 'gene_mean' values are greater than min_mean_val_for_zscore
    gene_mean_vector = (df['gene_mean'] > min_mean_val_for_zscore).values
    return max_mean_nondup_vector, gene_mean_vector

def get_gene_symbols_in_gene_stat_uniq_bool_file(gene_stat_uniq_bool_file):
    # Read the TSV file
    df = pd.read_csv(gene_stat_uniq_bool_file, sep='\t')
    gene_symbols_in_gene_stat_uniq_bool_file = np.array(df['gene_symbol'])
    return gene_symbols_in_gene_stat_uniq_bool_file

def get_human_fake_nonhuman_dataset(chunk_idx, mask_fraction):
    human_dataset = get_mixed_dataset(chunk_idx*2, mask_fraction, dataset_to_get="human", add_prefix_to_gene=False)
    fake_nonhuman_dataset = get_mixed_dataset(chunk_idx*2+1, mask_fraction, dataset_to_get="human", add_prefix_to_gene=True)
    human_fake_nonhuman_dataset = MixedDataset(human_dataset, fake_nonhuman_dataset)
    return human_fake_nonhuman_dataset

def get_mixed_dataset(chunk_idx, mask_fraction, dataset_to_get="both", add_prefix_to_gene=False):
    if dataset_to_get not in ["both", "human", "nonhuman"]:
        print('dataset_to_get has to be one of ["both", "human", "nonhuman"]')
        return None
    mouse_dataset, human_dataset, hm_dataset = None, None, None
    original_species_env_var = os.environ.get('SPECIES', None)
    if dataset_to_get in ["both", "nonhuman"]:
        os.environ['SPECIES'] = "Mouse"
    else:
        os.environ['SPECIES'] = "Human"
    from utils.json_utils import JsonUtils
    from train.common_params_funs import get_gene2idx
    from data.GN_Dataset import GN_Dataset
    from utils.config_loader import Config
    from utils.utils import get_gene_symbols_from_h5

    def get_dataset(
                    ARCHS_gene_expression_h5_path, 
                    gene_stat_uniq_bool_file,
                    output_file_prefix, 
                    chunk_idx, 
                    gene_to_idx,  
                    mask_fraction,
                    min_mean_val_for_zscore, 
                    expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD,
                    add_prefix_to_gene=add_prefix_to_gene
                    ):
        
        if expr_discretization_method == "uniform_bin_count_keep_ones":
            print(f'loading {output_file_prefix}_bin_tot2000_final_all_genes_chunk_{chunk_idx}.npy')
            ori_expression_data = np.load(f'{output_file_prefix}_bin_tot2000_final_all_genes_chunk_{chunk_idx}.npy')
        gene_symbols_from_h5 = get_gene_symbols_from_h5(ARCHS_gene_expression_h5_path)
        gene_symbols_in_gene_stat_uniq_bool_file = get_gene_symbols_in_gene_stat_uniq_bool_file(gene_stat_uniq_bool_file)
        are_identical = np.array_equal(gene_symbols_from_h5, gene_symbols_in_gene_stat_uniq_bool_file)
        assert are_identical, "The arrays are not identical."
        assert ori_expression_data.shape[1] == len(gene_symbols_from_h5), f"Number of genes in the h5 file diff from {output_file_prefix}_bin_tot2000_final_all_genes_chunk_{chunk_idx}.npy"
        genes_in_gene2vec_bool = [gene in gene_to_idx for gene in gene_symbols_from_h5]
        if os.environ['SPECIES'] == "Mouse" and params.GENE_EMB_NAME.startswith("hm_"):
            gene_prefix = config.get("gene_prefix")
            genes_in_gene2vec_bool = [f"{gene_prefix}{gene}" in gene_to_idx for gene in gene_symbols_from_h5]
            print(np.sum(np.array(genes_in_gene2vec_bool)))
        non_dup_gene_bool, qualified_mean_bool = get_non_dup_and_qualified_mean_bool_vectors(gene_stat_uniq_bool_file, min_mean_val_for_zscore = min_mean_val_for_zscore)
        
        in_gene2vec_nondup_qualified_mean_bool = genes_in_gene2vec_bool & non_dup_gene_bool & qualified_mean_bool
        idx_genes_in_emb_meet_z_dup = np.where(in_gene2vec_nondup_qualified_mean_bool)[0]
        gene_symbols_for_dataset_input = gene_symbols_from_h5[idx_genes_in_emb_meet_z_dup]
        sample_by_gene_expr_mat_for_dataset_input = ori_expression_data[:, idx_genes_in_emb_meet_z_dup]
        del ori_expression_data
        if (os.environ['SPECIES'] == "Mouse" and params.GENE_EMB_NAME.startswith("hm_")) or add_prefix_to_gene:
            if add_prefix_to_gene:
                gene_prefix = "m_"
            gene_symbols_for_dataset_input = [f"{gene_prefix}{gene}" for gene in gene_symbols_for_dataset_input]
        gn_dataset = GN_Dataset(
            sample_by_gene_expr_mat=sample_by_gene_expr_mat_for_dataset_input,
            gene_symbols=gene_symbols_for_dataset_input,
            n_bins=params.NUM_BINS, 
            mask_fraction=mask_fraction, 
            expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD, 
            num_of_genes=params.NUM_OF_GENES_SELECTED,
            number_of_special_embeddings=params.NUMBER_OF_SPECIAL_TOKEN_IN_DATASET,
            sort_return_expr_numerically=(params.TRANSFORMER_MODEL_NAME == "GPT" or params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens")
        )
        return gn_dataset
    
    if dataset_to_get in ["both", "nonhuman"]:
        config = Config()
        proj_path = config.proj_path
        print(proj_path)
        ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
        ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")
        gene_to_idx, _ = get_gene2idx()
        use_and_keep_zero_expr_genes = params.USE_AND_KEEP_ZERO_EXPR_GENES
        if use_and_keep_zero_expr_genes:
            output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
        else:
            output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"

        min_mean_val_for_zscore = params.MIN_MEAN_VAL_FOR_ZSCORE
        gene_stat_uniq_bool_file = output_file_prefix + ".gene_stat_filt_on_z_dup.tsv"

        mouse_dataset = get_dataset(
                    ARCHS_gene_expression_h5_path, 
                    gene_stat_uniq_bool_file,
                    output_file_prefix, 
                    chunk_idx, 
                    gene_to_idx,  
                    mask_fraction,
                    min_mean_val_for_zscore, 
                    expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD, 
                    )
        
        print("input parameters for getting mouse dataset: ")
        print((ARCHS_gene_expression_h5_path, 
                gene_stat_uniq_bool_file,
                output_file_prefix, 
                chunk_idx, 
                mask_fraction,
                min_mean_val_for_zscore))
        print(list(gene_to_idx.keys())[-20:])

    if dataset_to_get in ["both", "human"]:
        # Set environment variable
        os.environ['SPECIES'] = "Human"

        import importlib
        # Reload utils.config_loader
        import utils.config_loader
        importlib.reload(utils.config_loader)
        config = Config()
        print(config.proj_path)

        # Reload utils.json_utils
        import utils.json_utils
        importlib.reload(utils.json_utils)

        # Reload train.common
        import train.common
        importlib.reload(train.common)

        # Reload data.GN_Dataset
        import data.GN_Dataset
        importlib.reload(data.GN_Dataset)

        # Reload utils.utils
        import utils.utils
        importlib.reload(utils.utils)

        ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")
        ARCHS_gene_expression_h5_path = config.get("ARCHS_gene_expression_h5_path")
        gene_to_idx, _ = get_gene2idx()
        use_and_keep_zero_expr_genes = params.USE_AND_KEEP_ZERO_EXPR_GENES
        if use_and_keep_zero_expr_genes:
            output_file_prefix = config.proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
        else:
            output_file_prefix = config.proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"

        min_mean_val_for_zscore = params.MIN_MEAN_VAL_FOR_ZSCORE
        gene_stat_uniq_bool_file = output_file_prefix + ".gene_stat_filt_on_z_dup.tsv"

        human_dataset = get_dataset(
                        ARCHS_gene_expression_h5_path, 
                        gene_stat_uniq_bool_file,
                        output_file_prefix, 
                        chunk_idx, 
                        gene_to_idx,  
                        mask_fraction,
                        min_mean_val_for_zscore, 
                        expr_discretization_method=params.EXPR_DISCRETIZATION_METHOD, 
                        )
        print("input parameters for getting human dataset: ")
        print((ARCHS_gene_expression_h5_path, 
                gene_stat_uniq_bool_file,
                output_file_prefix, 
                chunk_idx, 
                mask_fraction,
                min_mean_val_for_zscore))
        print(list(gene_to_idx.keys())[:10])
    if dataset_to_get == "both":
        # Set the seed before creating the dataset
        MixedDataset.set_seed(8)  # Replace 42 with your desired seed

        # Assuming dataset1 and dataset2 are your existing Dataset objects
        hm_dataset = MixedDataset(human_dataset, mouse_dataset)
    if original_species_env_var is None:
        del os.environ['SPECIES']
        print(f"os.environ SPECIES now is deleted, as originally it wasn't set.")
    else:
        os.environ['SPECIES'] = original_species_env_var
        print(f"os.environ now is {os.environ['SPECIES']}.")
    
    if dataset_to_get == "both":
        return hm_dataset
    elif dataset_to_get == "human":
        return human_dataset
    elif dataset_to_get == "nonhuman":
        return mouse_dataset
