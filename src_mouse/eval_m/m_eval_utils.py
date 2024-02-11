import numpy as np
import pandas as pd
from utils.params import params
from sklearn.metrics.pairwise import cosine_similarity
from eval_m.m_visual_utils import plot_PCA, plot_histogram
from data_m.get_homolo_mg import get_homolo_mg
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
from dna.dna_utils import file_to_dict


# Sup. Alig.
# Unsup. Alig.
# SharedEmb

def reorder_symmetric_matrix(matrix, original_order, new_order):
    """
    Efficiently reorder the rows and columns of a 2D symmetric matrix according to a new order of its row/column names,
    using vectorized operations for better performance with large matrices.
    
    Parameters:
    matrix (np.array): The original 2D symmetric matrix to be reordered.
    original_order (list): The list of row/column names in the original order of the matrix.
    new_order (list): The list of row/column names in the desired new order.
    
    Returns:
    np.array: The reordered 2D symmetric matrix.
    """
    assert matrix.shape[0] == len(original_order), "matrix.shape[0] != len(original_order)"
    assert matrix.shape[1] == len(original_order), "matrix.shape[1] != len(original_order)"
    # Create a mapping from gene names to their index in the original matrix
    gene_to_index = {gene: idx for idx, gene in enumerate(original_order)}
    
    # Convert the new order into indices, with -1 for genes not in the original order
    new_indices = [gene_to_index.get(gene, -1) for gene in new_order]
    
    # Create a mask for valid indices (excluding -1)
    valid_mask = np.array(new_indices) != -1
    
    # Filter out valid indices for both rows and columns
    valid_row_indices = np.array(new_indices)[valid_mask]
    valid_col_indices = valid_row_indices.copy()  # Symmetric matrix has the same row and column indices

    # Initialize a new matrix filled with NaN
    new_matrix = np.full((len(new_order), len(new_order)), np.nan)
    
    # Use valid indices to fill the new matrix from the original matrix
    new_matrix[np.ix_(valid_mask, valid_mask)] = matrix[np.ix_(valid_row_indices, valid_col_indices)]

    return new_matrix

def mean_and_std_of_arrays(arrays):
    """
    Calculate the mean and standard deviation of a list of 2D arrays, handling missing values (NaNs),
    and returning NaN for elements that are all-NaN across all slices.
    
    Parameters:
    arrays (list of np.array): A list of 2D numpy arrays with the same shape.
    
    Returns:
    tuple: A tuple containing two 2D arrays, the first with mean values and the second with standard deviation values.
           Both arrays will have NaN where all elements were NaN in the input arrays.
    """
    # Convert the list of arrays to a 3D numpy array where each 2D array is a slice
    arr_stack = np.array(arrays)
    
    # Count the number of non-NaN elements along the new axis (0)
    count_non_nan = np.count_nonzero(~np.isnan(arr_stack), axis=0)
    
    # Calculate the mean, handling divisions by zero by returning NaN in those cases
    mean_arr = np.nanmean(arr_stack, axis=0)
    
    # Calculate standard deviation, using nanstd to handle NaNs properly
    std_arr = np.nanstd(arr_stack, axis=0)
    
    # Free up memory by deleting the 3D array
    del arr_stack
    
    return mean_arr, std_arr

def get_n_closest_tokens(similarity_matrix, gene_names, token_types, output_file, n_closest=10):
    """
    Selects the n closest tokens (genes) for each token based on the similarity matrix
    and outputs the results to a file along with the similarity values.
    It only considers tokens of a different type for the closest ones.

    Parameters:
    similarity_matrix (np.array): 2D array representing the similarity between tokens.
    gene_names (list): List of gene (token) IDs.
    token_types (list): List indicating the type of each token ('H' or 'M'), with the same length as idx2gene.
    n_closest (int): The number of closest tokens to retrieve for each token.
    output_file (str): Path to the output file where the results will be saved.
    """
    assert similarity_matrix.shape[0] == len(gene_names) == len(token_types), "Input lengths must match."
    
    # Initialize an array to hold the closest token IDs and their values
    closest_tokens_info = []
    
    # Iterate over each token to find the closest tokens of a different type
    for i, (token_id, token_type) in enumerate(zip(gene_names, token_types)):
        if i > 500000:
            break
        # Mask for selecting only tokens of a different type
        different_type_mask = np.array(token_types) != token_type
        
        # Mask the similarity matrix to consider only tokens of a different type
        masked_similarity = np.copy(similarity_matrix[i])
        masked_similarity[~different_type_mask] = -np.inf  # Assign a low value to same type tokens
        
        # Find the closest tokens indices and values of different type
        closest_indices = np.argsort(masked_similarity)[-n_closest:][::-1]
        closest_values = np.sort(masked_similarity)[-n_closest:][::-1]
        # Convert indices to actual token IDs and format the information
        closest_info = [f"{gene_names[idx]},{val:.4f}" for idx, val in zip(closest_indices, closest_values)]

        # Append the information to the list
        closest_tokens_info.append([token_id] + closest_info)
    
    # Convert the information list to a DataFrame
    column_names = ['Token_ID'] + [f'Closest_{i}' for i in range(1, n_closest+1)]
    df_closest_tokens = pd.DataFrame(closest_tokens_info, columns=column_names)

    # Save to TSV file
    df_closest_tokens.to_csv(output_file, sep='\t', index=False)
    print(f"Closest tokens saved to {output_file}")
    return df_closest_tokens





def find_rank_of_similarity(similarity_matrix, token_ids, ground_truth_mapping, token_types=None):
    """
    For each key in the ground truth mapping dictionary, find the rank of the similarity of the value.
    
    :param similarity_matrix: A 2D numpy array where each element [i, j] is the similarity between token i and token j.
    :param token_ids: A list of token IDs that correspond to the indices in the similarity matrix.
    :param ground_truth_mapping: A list of 2 element tuples, or dictionary where keys are token IDs in one language, and values are the corresponding token IDs in the other language.
    :param token_types: An optional list of the same length as token_ids indicating the type of each token.
    :return: Two dictionaries of ranks, one with only the first key gene as key, the second dict use gene pair as key. The order of the second one is same as input list or dict
    """
    ranks = {}
    ranks_gene_pair_as_key = {}
    # Creating a mapping from token ID to index in the similarity matrix
    id_to_index = {token_id: index for index, token_id in enumerate(token_ids)}

    if token_types is not None:
        # Convert token_types list to a numpy array for efficient operations
        token_types_array = np.array(token_types)

    if isinstance(ground_truth_mapping, dict):
        mappings = ground_truth_mapping.items()
    else:  # Assume it's a list of tuples if not a dict
        mappings = ground_truth_mapping

    for key_token_id, true_value_token_id in mappings:
        # Convert token IDs to indices in the similarity matrix
        key_index = id_to_index.get(key_token_id, None)
        true_value_index = id_to_index.get(true_value_token_id, None)
        if key_index is None or true_value_index is None:
            ranks_gene_pair_as_key[f"{key_token_id} {true_value_token_id}"] = np.nan
            ranks[key_token_id] = np.nan 
            continue

        # Get all similarity scores for the token corresponding to 'key_index'
        all_similarities = similarity_matrix[key_index].copy()
        
        # If token_types is provided, mask out similarities where the token is of the same type as key_token
        if token_types is not None:
            key_type = token_types[key_index]
            # Use numpy boolean array for faster masking
            mask = (token_types_array == key_type)
            all_similarities[mask] = -np.inf  # Set to -inf to ignore these similarities

        # Count how many similarities are greater than the true similarity
        true_similarity = similarity_matrix[key_index, true_value_index]
        rank = (all_similarities > true_similarity).sum() + 1  # Adding 1 because rank starts from 1
        
        # Save the rank to the ranks dictionary
        ranks[key_token_id] = rank
        ranks_gene_pair_as_key[f"{key_token_id} {true_value_token_id}"] = rank
    return ranks, ranks_gene_pair_as_key

def dict_to_file(dict_data, file_name):
    """
    Writes a dictionary to a text file, with each key-value pair on a new line,
    and the key and value separated by a space or tab.

    Parameters:
    dict_data (dict): The dictionary to write to the file.
    file_name (str): The name of the file to which to write the dictionary.
    """
    with open(file_name, 'w') as file:
        for key, value in dict_data.items():
            file.write(f"{key}\t{value}\n")
    print(f"Saved to {file_name}")

def get_similarity(name1, name2, similarity_matrix, id2idx):
    """
    get id2idx by this, and provide this to the function:
    id2idx = {name: idx for idx, name in enumerate(hm_gene_names)}
    """
    # Ensure the names are in the id2idx dictionary
    if name1 in id2idx and name2 in id2idx:
        # Get indices of the names using the dictionary
        index1 = id2idx[name1]
        index2 = id2idx[name2]
        # Return the similarity value
        return similarity_matrix[index1, index2]
    else:
        return np.nan  # Return NaN if names not found

def read_tsv_to_dict(file_path, key_col_idx=0, value_col_idx=1):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None)
    # Convert the DataFrame to a dictionary with the first column as keys and the second column as values
    return pd.Series(df[value_col_idx].values, index=df[key_col_idx]).to_dict()
def read_rank_dict(exp, epoch, config):
    out_dir = config.proj_path + f"/results/anal/{exp}/{exp}.epoch{epoch}.full_supervised"
    tsv_file = f"{out_dir}/m2h_ranks.tsv"
    m2h_rank_dict = read_tsv_to_dict(tsv_file)
    tsv_file = f"{out_dir}/h2m_ranks.tsv"
    h2m_rank_dict = read_tsv_to_dict(tsv_file)
    return h2m_rank_dict, m2h_rank_dict

def eval_simi_mat(simi_mat, gene_names, njson, config, output_prefix=None, is_pseudo_non_human=False):
    """
    dependence: get_homolo_mg, file_to_dict, find_rank_of_similarity, plot_histogram, get_similarity, np
    Gene names have to contain species prefix
    return rank_simi_results_dicts_for_gene_pair_of_interest, with structure of {gene_pair_label: (gene_pair_rank_result, gene_pair_similarity)}
    """

    gene_types = [njson.m_label if config.gene_prefix in gene else njson.h_label for gene in gene_names]
    id2idx = {gene: i for i, gene in enumerate(gene_names)}
    
    if is_pseudo_non_human:
        # h2m_one2one shouldn't contain any gene species prefix
        h2m_one2one = {gene: gene for gene in gene_names if "SPECIAL_EMB" not in gene and config.gene_prefix not in gene}
        m2h_one2one = h2m_one2one
        h2m_genes = {gene: [gene] for gene in h2m_one2one}
        m2h_genes = h2m_genes
    else:
        h2m_genes, m2h_genes, h2m_one2one, m2h_one2one = get_homolo_mg()
    h2m_one2one_added_prefix = {k: f"{config.gene_prefix}{v}" for k, v in h2m_one2one.items()}
    
    random_gene_pairs = file_to_dict(f"{config.proj_path}/results/anal/dna/random_gene_pairs/random_gene_pairs_not_have_close_emb_non_homolog.exp16_hm_BERT_coding.seed8.txt")
    random_gene_pairs = {k: config.gene_prefix+v for k, v in random_gene_pairs.items()}
    dict_of_gene_pair_dicts_of_interest = {
        njson.h2m_one2one: h2m_one2one_added_prefix,
        njson.random_gene_pairs: random_gene_pairs
    }
    rank_simi_results_dicts_for_gene_pair_of_interest = {}
    for gene_pair_label, gene_pair_dict_of_interest in dict_of_gene_pair_dicts_of_interest.items():
        _, gene_pair_dict_of_interest_rank_result = find_rank_of_similarity(simi_mat, gene_names, gene_pair_dict_of_interest, gene_types)
        plot_histogram(np.array(list(gene_pair_dict_of_interest_rank_result.values())),
                       figsize=(njson.one3rd_fig_width, njson.fig_height_a0),
                       log_scale=True,
                       output_file=f"{output_prefix}.{gene_pair_label}.ranks.hist.pdf",
                        xlabel=njson.rank_of_orthologs,
                        ylabel=njson.count_of_genes
                      )
        
        gene_pair_dict_of_interest_similarity = {f"{human_gene} {mouse_gene}": get_similarity(human_gene, mouse_gene, simi_mat, id2idx) for human_gene, mouse_gene in gene_pair_dict_of_interest.items()}
        
        plot_histogram(np.array(list(gene_pair_dict_of_interest_similarity.values())),
                       figsize=(njson.one3rd_fig_width, njson.fig_height_a0),
                       output_file=f"{output_prefix}.{gene_pair_label}.similarity.hist.pdf",
                        xlabel=njson.ortholog_similarity,
                        ylabel=njson.count_of_genes
                      )
        rank_simi_results_dicts_for_gene_pair_of_interest[gene_pair_label] = (gene_pair_dict_of_interest_rank_result, gene_pair_dict_of_interest_similarity)
    return rank_simi_results_dicts_for_gene_pair_of_interest
    
def eval_specified_emb(hm_emb, hm_gene_names, hm_token_types, config, njson, output_prefix=None, is_pseudo_non_human=False, pca_title=''):
    
    plot_PCA(hm_emb, 
             hm_token_types,
             figsize=(njson.one3rd_fig_width, njson.fig_height_a0),
             output_file=f"{output_prefix}.MUSE_emb_PCA.pdf",
             title=pca_title
            )
    similarity_matrix = cosine_similarity(hm_emb)
    return eval_simi_mat(similarity_matrix, hm_gene_names, njson, config, output_prefix=output_prefix, is_pseudo_non_human=is_pseudo_non_human)

def eval_MUSE_emb(dir_without_uid, params, config, njson, output_prefix=None, is_pseudo_non_human=False, pca_title=''):
    from train.common_params_funs import get_gene2idx
    gene2idx, _ = get_gene2idx(params.GENE_EMB_NAME)
    gene_to_ori_case_dict = get_gene_to_ori_case_dict(gene2idx)
    muse_result_dir = get_muse_result_uid_dir(dir_without_uid)
    hm_emb, hm_gene_names, hm_token_types = load_muse_result(muse_result_dir, gene_to_ori_case_dict=gene_to_ori_case_dict)
    plot_PCA(hm_emb, 
             hm_token_types,
             figsize=(njson.one3rd_fig_width, njson.fig_height_a0),
             output_file=f"{output_prefix}.MUSE_emb_PCA.pdf",
             title=pca_title
            )
    similarity_matrix = cosine_similarity(hm_emb)
    return eval_simi_mat(similarity_matrix, hm_gene_names, njson, config, output_prefix=output_prefix, is_pseudo_non_human=is_pseudo_non_human)

def n_closest_tokens_to_gene_pair_list(n_closest_tokens, config):
    n_closest_tokens_gene_pairs = []
    for index, row in n_closest_tokens.iterrows():
        gene1 = row['Token_ID']
        gene2 = row['Closest_1'].split(',')[0]
        if config.gene_prefix in gene1:
            human_gene = gene2
            mouse_gene = gene1
        else:
            human_gene = gene1
            mouse_gene = gene2
        gene_pair = f"{human_gene} {mouse_gene.replace(config.gene_prefix, '')}"
        if gene_pair not in n_closest_tokens_gene_pairs:
            n_closest_tokens_gene_pairs.append(gene_pair)
    return n_closest_tokens_gene_pairs