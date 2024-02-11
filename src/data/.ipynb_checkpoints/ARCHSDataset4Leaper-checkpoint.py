from utils.params import params
params = params()
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.json_utils import JsonUtils
ju = JsonUtils()
from train.functions_leaper import *

from utils.config_loader import Config
config = Config()
proj_path = config.proj_path
ARCHS_file_basename_prefix = config.get("ARCHS_file_basename_prefix")

from data.discretize_expression import discretize_expression

# Load gene_to_idx dictionary from JSON file
gene_to_idx_path = config.get("gene2vec_gene_to_idx_json_path")
gene_to_idx = ju.load_data_from_file(gene_to_idx_path)
std_dev_for_another_normal_distr_for_binning = STD_DEV_FOR_ANOTHER_NORMAL_DISTR_FOR_BINNING
use_and_keep_zero_expr_genes = params.USE_AND_KEEP_ZERO_EXPR_GENES
if use_and_keep_zero_expr_genes:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_with_zero_expr_genes"
else:
    output_file_prefix = proj_path + f"/data/external/ARCHS/normalize_each_gene/{ARCHS_file_basename_prefix}_without_zero_expr_genes"


class ARCHSDataset(Dataset):
    def __init__(self, 
    h5_file_path, 
    n_bins=params.NUM_BINS, 
    mask_fraction=0.15, 
    expr_discretization_method="Direct_quantile", 
    load_data_into_mem=True, 
    chunk_idx=0, 
    num_of_genes=params.NUM_OF_GENES_SELECTED,
    gene_stat_uniq_bool_file=output_file_prefix + ".gene_stat_filt_on_z_dup.tsv",
    min_mean_val_for_zscore=params.MIN_MEAN_VAL_FOR_ZSCORE,
    only_use_postive_zscores_in_training=params.ONLY_USE_POSITIVE_ZSCORES_IN_TRAINING,
    max_sim_sample_per_sample=MAX_SIM_SAMPLE_PER_SAMPLE_FOR_LEAPER, 
    min_sim=MIN_SIMILARITY_FOR_LEAPER_SAMPLES, 
    max_sim=MAX_SIMILARITY_FOR_LEAPER_SAMPLES, 
    number_of_special_embeddings=NUMBER_OF_SPECIAL_TOKEN
    ):
        self.h5_file_path = h5_file_path
        self.h5_file = h5py.File(self.h5_file_path, "r")
        self.gene_symbols = self.h5_file['meta']['genes']['symbol'][()]
        self.sample_size = self.h5_file['data']['expression'].shape[1]
        self.n_bins = n_bins
        self.num_of_genes_to_return = num_of_genes
        # stat for the mean and std for each gene, and a boolean to indicate if that genes should be selected 
        # to ensure the uniqueness
        self.gene_stat_uniq_bool_file = gene_stat_uniq_bool_file
        self.min_mean_val_for_zscore = min_mean_val_for_zscore
        self.only_use_postive_zscores_in_training = only_use_postive_zscores_in_training
        # Find the index of genes in gene_to_idx
        self.gene_symbol_indices = np.array([gene_to_idx.get(gene_symbol.decode('utf-8'), -1) for gene_symbol in self.gene_symbols], dtype=np.int32)
        self.mask_fraction = mask_fraction
        # Filter gene_symbol_indices to keep only the genes with index in gene_to_idx
        self.genes_in_gene2vec = self.gene_symbol_indices != -1
        # we put it here, as we would like to make treating it a hyperparam easier, 
        # we don't have to go back to generate the gene_stat_uniq_bool_file file, if we do the filtering, 
        # before generating that file. We would like to remove the genes that have very small means,
        # which can make the normally expressed genes in this dataset have very high z-score
        self.non_dup_gene_bool, self.qualified_mean_bool = self.get_non_dup_and_qualified_mean_bool_vectors()
        self.in_gene2vec_nondup_bool = self.genes_in_gene2vec & self.non_dup_gene_bool
        self.in_gene2vec_nondup_qualified_mean_bool = self.in_gene2vec_nondup_bool & self.qualified_mean_bool

        # df = pd.DataFrame({
        #     'genes_in_gene2vec': self.genes_in_gene2vec,
        #     'non_dup_gene_bool': self.non_dup_gene_bool,
        #     'qualified_mean_bool': self.qualified_mean_bool,
        #     'in_gene2vec_nondup_bool': self.in_gene2vec_nondup_bool,
        #     'in_gene2vec_nondup_qualified_mean_bool': self.in_gene2vec_nondup_qualified_mean_bool
        # })

        # # Export the DataFrame to a TSV file
        # df.to_csv(proj_path + '/data/external/ARCHS/normalize_each_gene/tmp.output.tsv', sep='\t', index=False)

        # zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
        # so length of self.qualified_mean_bool_for_npy_file will match with the dimension of final npy file
        self.qualified_mean_bool_for_npy_file = self.in_gene2vec_nondup_qualified_mean_bool[self.in_gene2vec_nondup_bool]
        self.filtered_indices = np.where(self.in_gene2vec_nondup_qualified_mean_bool)[0]
        self.gene_indices_for_ret = torch.tensor(self.gene_symbol_indices[self.in_gene2vec_nondup_qualified_mean_bool], dtype=torch.int32)
        #raw integer read count, genes are rows, samples are columns
        self.h5_file_data_expr = self.h5_file['data']['expression']
        self.expr_discretization_method = expr_discretization_method
        #the rows are samples, it is transposed from the original matrix
        #zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
        print(output_file_prefix + f"_final_chunk_{chunk_idx}.npy")
        self.filtered_expression_data = np.load(output_file_prefix + f"_final_chunk_{chunk_idx}.npy")
        self.filtered_expression_data = np.load(f'{h5_file_path}.filtered_expression_data_chunk_{chunk_idx}.npy')
        self.sim_samples = self.select_similar_samples(f'{h5_file_path}.filtered_expression_data_chunk_{chunk_idx}.sample_sim.csv',
                                                       max_sample=max_sim_sample_per_sample, 
                                                       min_sim=min_sim, 
                                                       max_sim=max_sim)
        self.data_len = np.array(self.sim_samples).shape[0]
        self.number_of_special_embeddings = number_of_special_embeddings
            
    def get_non_dup_and_qualified_mean_bool_vectors(self):
        # Read the TSV file
        df = pd.read_csv(self.gene_stat_uniq_bool_file, sep='\t')

        # Create a numpy boolean vector from the 'max_mean_nondup' column
        max_mean_nondup_vector = df['max_mean_nondup'].values.astype(bool)

        # Create a numpy boolean vector that indicates whether the 'gene_mean' values are greater than min_mean_val_for_zscore
        gene_mean_vector = (df['gene_mean'] > self.min_mean_val_for_zscore).values
        return max_mean_nondup_vector, gene_mean_vector
        
    def __len__(self):
        return self.data_len
    def prepend_special_tokens(self, result, number_of_special_embeddings):
        # Prepend indices for gene_indices
        special_tokens_indices = torch.arange(number_of_special_embeddings, dtype=result['gene_indices'].dtype)
        result['gene_indices'] = torch.cat([special_tokens_indices, result['gene_indices']])
        
        # Prepend zeros for input_binned_expr and output_binned_expr
        zeros_to_prepend = torch.zeros(number_of_special_embeddings, dtype=result['input_binned_expr'].dtype)
        result['input_binned_expr'] = torch.cat([zeros_to_prepend, result['input_binned_expr']])
        result['output_binned_expr'] = torch.cat([zeros_to_prepend, result['output_binned_expr']])
        
        # Prepend False for zero_expression_genes
        false_to_prepend = torch.zeros(number_of_special_embeddings, dtype=torch.bool)
        result['input_zero_expression_genes'] = torch.cat([false_to_prepend, result['input_zero_expression_genes']])
        result['output_zero_expression_genes'] = torch.cat([false_to_prepend, result['output_zero_expression_genes']])
        
        return result
        
    def get_ori_exp(self, idx):
        expression_vector = self.h5_file_data_expr[self.filtered_indices, idx]
        return expression_vector
    
    def __getitem__(self, idx):
        if self.num_of_genes_to_return <= 0:
            return self.__getitem_all_genes_(idx)
        else:
            return self.__getitem_only_some_genes__(idx)
    
    def __getitem_all_genes_(self, idx):
        #start_time = time.time()
        #most time spends on this single line below:
        if self.load_data_into_mem == True:
            #the rows are samples, it is transposed from the original matrix
            #zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
            expression_vector = self.filtered_expression_data[idx, self.qualified_mean_bool_for_npy_file]
        else:
            expression_vector = self.h5_file_data_expr[self.filtered_indices, idx]
            #print(f"self.h5_file time: {time.time() - start_time:.4f} seconds")
            expression_vector = np.array(expression_vector).T  # Transpose the expression vector
        
        #print(expression_vector.shape)
        #print(f"T time: {time.time() - start_time:.4f} seconds")
        # Convert expression_vector to a numpy array with dtype float32
        expression_vector_float32 = expression_vector.astype(np.float32)
        #print(f"np.float32 time: {time.time() - start_time:.4f} seconds")
        # Discretize expression_vector_float32 into 100 bins
        if self.expr_discretization_method == "Direct_quantile":
            discretized_expression, zero_expression_genes = discretize_expression(expression_vector_float32, self.n_bins, std_dev=std_dev_for_another_normal_distr_for_binning)
        #print(f"discretized_expression time: {time.time() - start_time:.4f} seconds")
        mask = np.random.rand(discretized_expression.size) < self.mask_fraction
        #print(f"np.random.rand time: {time.time() - start_time:.4f} seconds")

        masked_expression = discretized_expression.copy()
        #print(f"copy time: {time.time() - start_time:.4f} seconds")
        masked_expression[mask] = 0

        return {
            "gene_indices": self.gene_indices_for_ret,
            "masked_expression": torch.tensor(masked_expression, dtype=torch.int32),
            "true_expression": torch.tensor(discretized_expression, dtype=torch.int32),
            "zero_expression_genes": torch.tensor(zero_expression_genes, dtype=torch.bool)
        }
    # input numpy array
    # output numpy arrays of top genes, and the numeric indices
    def get_top_genes(self, expression_vector_float32, num_of_genes_to_return, only_use_postive_zscores_in_training):
        # Get indices of top highest absolute values
        if only_use_postive_zscores_in_training:
            abs_expr = expression_vector_float32
        else:
            abs_expr = np.abs(expression_vector_float32)
        top_num = -1 * num_of_genes_to_return
        top_abs_indices = np.argpartition(abs_expr, top_num)[top_num:]

        # Keep only the top highest absolute values
        expression_vector_float32 = expression_vector_float32[top_abs_indices]
        return expression_vector_float32, top_abs_indices
        
    def __getitem_only_some_genes__(self, idx, top_abs_indices=None, return_top_abs_indices=False):
        #zscore final npy files only have in_gene2vec_nondup_bool, but no qualified_mean was applied
        expression_vector = self.filtered_expression_data[idx, self.qualified_mean_bool_for_npy_file]

        expression_vector_float32 = expression_vector.astype(np.float32)
        
        if top_abs_indices == None:
            expression_vector_float32, top_abs_indices = self.get_top_genes(expression_vector_float32, self.num_of_genes_to_return, self.only_use_postive_zscores_in_training)
        else:
            expression_vector_float32 = expression_vector_float32[top_abs_indices]
            
        if self.expr_discretization_method == "Direct_quantile":
            discretized_expression, zero_expression_genes = discretize_expression(expression_vector_float32, self.n_bins, std_dev=std_dev_for_another_normal_distr_for_binning)

        mask = np.random.rand(discretized_expression.size) < self.mask_fraction

        masked_expression = discretized_expression.copy()
        masked_expression[mask] = 0
        ret_item = {
            "gene_indices": self.gene_indices_for_ret[top_abs_indices],
            "masked_expression": torch.tensor(masked_expression, dtype=torch.int32),
            "true_expression": torch.tensor(discretized_expression, dtype=torch.int32),
            "zero_expression_genes": torch.tensor(zero_expression_genes, dtype=torch.bool)
        }
        if return_top_abs_indices:
            ret_item = (ret_item, top_abs_indices)
        return ret_item
        
    def __getitem__(self, idx):
        #start_time = time.time()
        #most time spends on this single line below:
        input_expr_idx = int(self.sim_samples[idx][0])
        output_expr_idx = int(self.sim_samples[idx][1])
        input_expr_item, top_abs_indices = self.__getitem_only_some_genes__(input_expr_idx, top_abs_indices=None, return_top_abs_indices=True)
        #make sure the selected genes are as same as input 
        output_expr_item = self.__getitem_only_some_genes__(output_expr_idx, top_abs_indices=top_abs_indices, return_top_abs_indices=False)
        result = {
            "input_output_idx": torch.tensor([input_expr_idx, output_expr_idx]),
            "gene_indices": input_expr_item["gene_indices"],
            "input_binned_expr": input_expr_item["masked_expression"],
            "output_binned_expr": output_expr_item["true_expression"],
            "input_zero_expression_genes": input_expr_item["zero_expression_genes"],
            "output_zero_expression_genes": output_expr_item["zero_expression_genes"]
        }
        return self.prepend_special_tokens(result, self.number_of_special_embeddings)

    def __del__(self):
        self.h5_file.close()

