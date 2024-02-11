import anndata
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.config_loader import Config
from utils.Ensembl_ID_gene_symbol_mapping_reader import read_ensembl_to_gene_mapping
from utils.json_utils import JsonUtils
from data.discretize_expression import discretize_expression
from utils.string_tensor import string_to_tensor
config = Config()
ju = JsonUtils()

# Your existing code and functions
def random_selection(input_list, num_elements):
    if num_elements >= len(input_list):
        return input_list
    else:
        return random.sample(input_list, num_elements)
class ReploglePerturbationDataset(Dataset):
    def __init__(
        self,
        adata,
        gene_to_idx_path=None,
        emsembl2gene_path=None,
        sample_number_for_each_perturbation=10,
        n_expr_bins=100,
        num_of_perturb_replicates=500,
        output_one_gene_every=1,
        perturbation_type="knockdown",
        expr_bin_for_fake_additional_gene_output=-1,
        label=None
    ):
        self.sample_number_for_each_perturbation = sample_number_for_each_perturbation
        #self.config = config
        #self.file_path_key_in_config = file_path_key_in_config
        self.n_expr_bins = n_expr_bins
        self.perturbation_type = perturbation_type
        self.expr_bin_for_fake_additional_gene_output = expr_bin_for_fake_additional_gene_output
        self.output_one_gene_every = output_one_gene_every
        self.gene_to_idx = ju.load_data_from_file(gene_to_idx_path) if gene_to_idx_path else None
        self.emsembl2gene = (
            read_ensembl_to_gene_mapping(emsembl2gene_path) if emsembl2gene_path else None
        )
        # Your existing dataset preparation code
        #self.adata = anndata.read_h5ad(config.get(file_path_key_in_config))
        self.adata = adata
        self.inf_count = 0
        self.neg_inf_count = 0
        self.nan_count = 0
        self.num_of_perturb_replicates = num_of_perturb_replicates
        cell_group_names = self.adata.obs_names
        gene_names = self.adata.var_names
        self.label = string_to_tensor(label)

        # the indices here are the indices of perturb cell groups (the the perturbed gene also embedable) in the original h5ad matrix
        indices_perturb = []
        # the indices here are the indices of baseline cell groups in the original h5ad matrix
        indices_baseline = []
        # the key here are the strings for matching the string at the position of gene_symbol in baseline cell groups
        baseline_keys = ["non-targeting"]
        #including the baseline and perturbed gene symbols, has the same length as original h5ad matrix
        perturbed_genes = []
        #indices all starts from zero
        for i, cell_group_name in enumerate(cell_group_names):
            ## to_change if other data set
            perturbed_gene = cell_group_name.split("_")[1]
            perturbed_genes.append(perturbed_gene)
            # if it is a baseline cell group, but not a perturbed cell group
            ## to_change if other data set
            if perturbed_gene in baseline_keys:
                indices_baseline.append(i)
            # only used perturbed genes that can be embedded for training
            elif perturbed_gene in self.gene_to_idx:
                indices_perturb.append(i)

        #only select the genes that have mapped gene symbols and have embedding index in gene_to_dix
        #the indices here are the indices of the selected genes in orginal h5ad file
        self.selected_indices = []
        #symbols of genes that have mapped gene symbols and have embedding index in gene_to_dix
        self.selected_gene_symbols = []
        #the indices here are the indices of the selected genes in geneID embedding
        self.indices_in_gene2vec = []
        found = 0
        for i, gene in enumerate(gene_names):
            #still works if gene is gene symbol but not ensembl ID
            gene_symbol = self.emsembl2gene.get(gene, gene)
            if gene_symbol in self.gene_to_idx:
                self.selected_gene_symbols.append(gene_symbol)
                self.selected_indices.append(i)
                self.indices_in_gene2vec.append(self.gene_to_idx[gene_symbol])
        if sample_number_for_each_perturbation > len(indices_baseline):
            sample_number_for_each_perturbation = len(indices_baseline)
        self.dataset_length = len(indices_baseline) + sample_number_for_each_perturbation * len(indices_perturb)

        self.dataset_metadata = {}
        i = 0
        #the indices all starts from zero
        for idx_baseline in indices_baseline:
            self.dataset_metadata[i] = {"idx_of_perturbation": None,
                                    "idx_of_baseline_for_this_perturb": idx_baseline,
                                    "perturbed_gene": None,
                                    "perturbation_type": "baseline"
                                    }
            i += 1
        #the indices all starts from zero
        for idx_perturbation in indices_perturb:
            baseline_indices_for_this_perturb = random_selection(indices_baseline, sample_number_for_each_perturbation)
            for baseline_idx_for_this_perturb in baseline_indices_for_this_perturb:
                self.dataset_metadata[i] = {"idx_of_perturbation": idx_perturbation,
                                        "idx_of_baseline_for_this_perturb": baseline_idx_for_this_perturb,
                                        "perturbed_gene": perturbed_genes[idx_perturbation],
                                        "perturbation_type": perturbation_type
                                        }
                i += 1



    
    def __len__(self):
        return self.dataset_length
    
    ## to_change if other data set doesn't use inf, -inf, and NaN
    def sanitize_input_raw_expr(self, input_raw_expr):
        # Find indices of inf, -inf, and NaN values
        inf_indices = np.isinf(input_raw_expr)
        neg_inf_indices = np.isneginf(input_raw_expr)
        nan_indices = np.isnan(input_raw_expr)

        # Temporarily replace inf and -inf with NaN to find the max and min finite values
        temp_input = input_raw_expr.copy()
        temp_input[np.logical_or(inf_indices, neg_inf_indices)] = np.nan
        max_value = np.nanmax(temp_input)
        min_value = np.nanmin(temp_input)

        # Replace inf, -inf, and NaN values
        input_raw_expr[inf_indices] = max_value
        input_raw_expr[neg_inf_indices] = min_value

        # Update the counters
        self.inf_count += np.sum(inf_indices)
        self.neg_inf_count += np.sum(neg_inf_indices)
        self.nan_count += np.sum(nan_indices)

        # Replace NaN values
        input_raw_expr[nan_indices] = np.random.uniform(min_value, max_value, size=np.sum(nan_indices))

        return input_raw_expr

    
    #to make sure the baseline cell groups have the same number of expression genes as perturbation cell groups
    def randomly_select_one_gene_and_append_to_3_list(self, 
                                                      indices_in_gene2vec, 
                                                      input_binned_expr, 
                                                      output_binned_expr, 
                                                      expr_bin_for_selected_gene_output = None):
        random_index = random.randint(0, len(indices_in_gene2vec) - 1)

        # Select elements from each list/array using the random index
        selected_index = indices_in_gene2vec[random_index]
        selected_input = input_binned_expr[random_index]
        selected_output = output_binned_expr[random_index]
        if expr_bin_for_selected_gene_output != None:
            selected_output = expr_bin_for_selected_gene_output

        # Copy the original list and append the selected element to the new list
        new_indices_in_gene2vec = indices_in_gene2vec.copy()
        new_indices_in_gene2vec.append(selected_index)

        # Append selected elements to the input_binned_expr and output_binned_expr arrays
        new_input_binned_expr = np.append(input_binned_expr, selected_input)
        new_output_binned_expr = np.append(output_binned_expr, selected_output)

        return new_indices_in_gene2vec, new_input_binned_expr, new_output_binned_expr

    def qc_data(self, input_raw_expr, idx):
        if np.isnan(input_raw_expr).any():
            print("Issue NA encountered at index:", idx)
            print("NA input_raw_expr:", input_raw_expr)
        if np.isinf(input_raw_expr).any():
            print("Issue INF encountered at index:", idx)
            print("INF input_raw_expr:", input_raw_expr)
    
    def select_one_out_of_every_some_genes(self, result, output_one_gene_every):
        selected_indices = list(range(0, result["gene_indices"].shape[0], output_one_gene_every))

        result["gene_indices"] = result["gene_indices"][selected_indices]
        result["input_binned_expr"] = result["input_binned_expr"][selected_indices]
        result["output_binned_expr"] = result["output_binned_expr"][selected_indices]

        return result

    def __getitem__(self, idx):
        #print(f"idx: {idx}")
        one_meta_data = self.dataset_metadata[idx]

        # Get the input and output expressions
        input_raw_expr = self.adata.X[one_meta_data["idx_of_baseline_for_this_perturb"], self.selected_indices]
        #print(f"input_raw_expr shape: {input_raw_expr.shape}")
        input_raw_expr = self.sanitize_input_raw_expr(input_raw_expr)
        input_binned_expr, input_zero_expression_genes = discretize_expression(input_raw_expr, self.n_expr_bins)

        true_expr_of_added_genes = np.full(self.num_of_perturb_replicates, self.expr_bin_for_fake_additional_gene_output)

        if one_meta_data["perturbation_type"] == "baseline":
            
            # Randomly select 100 genes for baseline cell lines
            random_indices = np.random.choice(len(input_binned_expr), self.num_of_perturb_replicates, replace=self.num_of_perturb_replicates >= len(input_binned_expr))
                # Get the random elements from self.indices_in_gene2vec and input_binned_expr using list comprehension
            random_gene_indices = [self.indices_in_gene2vec[i] for i in random_indices]
            random_input_binned_expr = [input_binned_expr[i] for i in random_indices]
            ret_gene_indices = torch.tensor(np.concatenate((self.indices_in_gene2vec, random_gene_indices)))
            #
            result = {
                "dataset_label": self.label,
                "gene_indices": ret_gene_indices,
                "input_binned_expr": torch.tensor(np.concatenate((input_binned_expr, random_input_binned_expr))),
                "output_binned_expr": torch.tensor(np.concatenate((input_binned_expr.copy(), true_expr_of_added_genes))),
                "zero_expression_genes": torch.zeros_like(ret_gene_indices, dtype=torch.bool)
            }
            if self.output_one_gene_every != 1:
                result = self.select_one_out_of_every_some_genes(result, self.output_one_gene_every)
            return result

        else:  # perturb cell group (sample)
            output_raw_expr = self.adata.X[one_meta_data["idx_of_perturbation"], self.selected_indices]
            output_raw_expr = self.sanitize_input_raw_expr(output_raw_expr)
            output_binned_expr, output_zero_expression_genes = discretize_expression(output_raw_expr, self.n_expr_bins)

            # Replicate perturbed genes by 100 times for perturbation cell lines
            perturbed_gene = one_meta_data["perturbed_gene"]
            perturbed_gene_index = self.gene_to_idx[perturbed_gene]
            perturbed_gene_indices = [perturbed_gene_index] * self.num_of_perturb_replicates

            # Adjust the expression of the perturbed gene
            if "knockdown" == one_meta_data["perturbation_type"].lower() or "knockoff" == one_meta_data["perturbation_type"].lower():
                expr_of_perturbed_gene = 1
            else:  # knockup
                expr_of_perturbed_gene = self.n_expr_bins

            perturbed_gene_expr = np.full(self.num_of_perturb_replicates, expr_of_perturbed_gene)
            
            ret_gene_indices = torch.tensor(np.concatenate((self.indices_in_gene2vec, perturbed_gene_indices)))
            #"dataset_label": self.label,
            result = {
                "dataset_label": self.label,
                "gene_indices": ret_gene_indices,
                "input_binned_expr": torch.tensor(np.concatenate((input_binned_expr, perturbed_gene_expr))),
                "output_binned_expr": torch.tensor(np.concatenate((output_binned_expr, true_expr_of_added_genes))),
                "zero_expression_genes": torch.zeros_like(ret_gene_indices, dtype=torch.bool)
            }
            if self.output_one_gene_every != 1:
                result = self.select_one_out_of_every_some_genes(result, self.output_one_gene_every)
            return result
        
    def __getitem_old__(self, idx):
        one_meta_data = self.dataset_metadata[idx]
        # baseline cell group (sample)
        if one_meta_data["perturbation_type"] == "baseline":
            input_raw_expr = self.adata.X[one_meta_data["idx_of_baseline_for_this_perturb"], self.selected_indices]
            input_raw_expr = self.sanitize_input_raw_expr(input_raw_expr)
            input_binned_expr = discretize_expression(input_raw_expr, self.n_expr_bins)
            output_binned_expr = input_binned_expr.copy()
            new_indices_in_gene2vec, new_input_binned_expr, new_output_binned_expr = self.randomly_select_one_gene_and_append_to_3_list(self.indices_in_gene2vec, 
                                                                                                                                   input_binned_expr, 
                                                                                                                                   output_binned_expr, 
                                                                                                                                   self.expr_bin_for_fake_additional_gene_output)
            result = {
                "gene_indices": torch.tensor(new_indices_in_gene2vec),
                "input_binned_expr": torch.tensor(new_input_binned_expr),
                "output_binned_expr": torch.tensor(new_output_binned_expr)
            }
            return result
        # perturb cell group (sample)
        else:
            input_raw_expr = self.adata.X[one_meta_data["idx_of_baseline_for_this_perturb"], self.selected_indices]
            input_raw_expr = self.sanitize_input_raw_expr(input_raw_expr)
            input_binned_expr = discretize_expression(input_raw_expr, self.n_expr_bins)
            output_raw_expr = self.adata.X[one_meta_data["idx_of_perturbation"], self.selected_indices]
            output_raw_expr = self.sanitize_input_raw_expr(output_raw_expr)
            output_binned_expr = discretize_expression(output_raw_expr, self.n_expr_bins)
            #if it is knockup, set it as the highest expr
            expr_of_perturbed_gene = self.n_expr_bins
            #if it is knockdown, set it as the lowest expr
            if "knockdown" == one_meta_data["perturbation_type"].lower() or "knockoff" == one_meta_data["perturbation_type"].lower():
                expr_of_perturbed_gene = 1
            #indices_in_gene2vec of all the genes for this cell group
            indices_in_gene2vec_for_this_cell_group = self.indices_in_gene2vec

            #the perturbed gene has expression in the original matrix
            if one_meta_data["perturbed_gene"] in self.selected_gene_symbols:
                #print("found")
                ##selected_gene_symbols are the symbols of the selected genes, which have mapped gene symbols and have embedding index in gene_to_dix
                idx_of_perturbed_gene_in_selected_mat = self.selected_gene_symbols.index(one_meta_data["perturbed_gene"])
                input_binned_expr[idx_of_perturbed_gene_in_selected_mat] = expr_of_perturbed_gene
                output_binned_expr[idx_of_perturbed_gene_in_selected_mat] = expr_of_perturbed_gene
                new_indices_in_gene2vec, new_input_binned_expr, new_output_binned_expr = self.randomly_select_one_gene_and_append_to_3_list(self.indices_in_gene2vec, 
                                                                                                                                       input_binned_expr, 
                                                                                                                                       output_binned_expr, 
                                                                                                                                       self.expr_bin_for_fake_additional_gene_output)
                return {
                    "gene_indices": torch.tensor(new_indices_in_gene2vec),
                    "input_binned_expr": torch.tensor(new_input_binned_expr),
                    "output_binned_expr": torch.tensor(new_output_binned_expr)
                }
            #the perturbed gene doesn't have expression in the original matrix
            else:
                #print("not found")
                idx_of_perturbed_gene_in_gene_id_embedding = self.gene_to_idx[one_meta_data["perturbed_gene"]]
                input_binned_expr = np.append(input_binned_expr, expr_of_perturbed_gene)
                output_binned_expr = np.append(output_binned_expr, expr_of_perturbed_gene)
                indices_in_gene2vec_for_this_cell_group = self.indices_in_gene2vec + [idx_of_perturbed_gene_in_gene_id_embedding]

                return {
                    "gene_indices": torch.tensor(indices_in_gene2vec_for_this_cell_group),
                    "input_binned_expr": torch.tensor(input_binned_expr),
                    "output_binned_expr": torch.tensor(output_binned_expr)
                }
    def __del__(self):
        #self.adata.file.close()
        print("Number of records with inf values:", self.inf_count)
        print("Number of records with -inf values:", self.neg_inf_count)
        print("Number of records with NaN values:", self.nan_count)



