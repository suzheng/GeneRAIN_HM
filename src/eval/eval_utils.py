
from utils.utils import get_model, get_config, get_gene2idx
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
import random
from collections import Counter
import copy
import torch
import glob
import csv
config = get_config()

def get_self_emb(param_json_file, check_point_path):
    from utils.params import params
    params = params()
    model = get_model(param_json_file, check_point_path)
    import os
    if 'PARAM_JSON_FILE' not in os.environ:
        os.environ['PARAM_JSON_FILE'] = param_json_file

    if params.TRANSFORMER_MODEL_NAME == "GPT":
        return model.gene_expr_transformer.gpt.transformer.wpe, model.gene_expr_transformer.gpt.transformer.wte
    elif params.TRANSFORMER_MODEL_NAME == "Bert_pred_tokens":
        if hasattr(model.gene_expr_transformer, 'bert_pred_tokens_model'):
            return model.gene_expr_transformer.bert_pred_tokens_model.bert.embeddings.position_embeddings, model.gene_expr_transformer.bert_pred_tokens_model.bert.embeddings.word_embeddings
        else:
            return model.gene_expr_transformer.bert_for_masked_lm_model.bert.embeddings.position_embeddings, model.gene_expr_transformer.bert_for_masked_lm_model.bert.embeddings.word_embeddings
    elif params.TRANSFORMER_MODEL_NAME == "BertExprInnerEmb":

        expr_emb = model.gene_expr_transformer.bert_for_masked_lm_model.bert.embeddings.word_embeddings
        gene_id_emb = model.gene_expr_transformer.bert_for_masked_lm_model.bert.embeddings.position_embeddings
        print(f"Swap the token and position embeddings for BertExprInnerEmb, as the Bert from the huggingface library predicts the tokens.")
        print(f"shape of the returned expr emb: {expr_emb.weight.shape}, gene id emb: {gene_id_emb.weight.shape}")
        return expr_emb, gene_id_emb
    else:
        return model.gene_expr_transformer.expression_emb, model.gene_expr_transformer.token_emb

def get_gf_model(get_geneformer_12L=False):
    import torch
    from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification
    import os
    config = get_config()
    model_directory = config.proj_path + "/data/external/models/Geneformer/"
    if get_geneformer_12L:
        print("get the geneformer_12L")
        model_directory = config.proj_path + "/data/external/models/Geneformer/geneformer-12L-30M/"
    gf_model = BertForMaskedLM.from_pretrained(model_directory, 
                                                    output_hidden_states=True, 
                                                    output_attentions=False)
    return gf_model
def get_gf_emb(gf_model):
    return gf_model.bert.embeddings.position_embeddings, gf_model.bert.embeddings.word_embeddings

def get_gf_gene2idx():
    config = get_config()
    def load_pickle(pickle_file):
        import pickle
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        return data
    token_dict = load_pickle(config.proj_path + "/data/external/models/Geneformer/geneformer/token_dictionary.pkl")
    gene_name_id_dict = load_pickle(config.proj_path + "/data/external/models/Geneformer/geneformer/gene_name_id_dict.pkl")
    gene_median_dict = load_pickle(config.proj_path + "/data/external/models/Geneformer/geneformer/gene_median_dictionary.pkl")
    ensemble2symbol = {v:k for k, v in gene_name_id_dict.items()}
    gf_gene2idx = {ensemble2symbol.get(k, k):v for k, v in token_dict.items()}
    return gf_gene2idx

def get_perturb_clusters_from_Replogle_etal():
    import pandas as pd
    config = get_config()
    data_file = config.proj_path + "/data/external/Replogle_etal/ScienceDirect_files_21Sep2023_03-18-32.332/1-s2.0-S0092867422005979-mmc3.perturbation_clusters.txt"
    df = pd.read_csv(data_file, sep='\t')
    df.rename(columns={"Unnamed: 0": "row"}, inplace=True)
    members_dict = {row['row']: row['members'].split(',') for _, row in df.iterrows()}
    nearby_genes_dict = {row['row']: row['nearby_genes'].split(',') for _, row in df.iterrows()}
    gene_to_cluster_dict = {gene: group for group, gene_list in members_dict.items() for gene in gene_list}
    return members_dict, nearby_genes_dict, gene_to_cluster_dict

def get_overlapping_genes(list1, list2):
    return list(set(list1) & set(list2))



def remove_genes_in_multiple_groups(gene_groups):
    # Flatten the list of genes and count the occurrences of each gene
    all_genes = [gene for genes in gene_groups.values() for gene in genes]
    gene_counts = Counter(all_genes)

    # Filter out genes that are present in two or more groups
    filtered_gene_groups = {group: [gene for gene in genes if gene_counts[gene] == 1] 
                            for group, genes in gene_groups.items()}

    # Optionally, remove groups that are now empty
    filtered_gene_groups = {group: genes for group, genes in filtered_gene_groups.items() if genes}

    return filtered_gene_groups

def emb_to_cluster_performance_metrics(emb_data, genes_of_the_emb_data, ground_true_gene_to_cluster_dict, n_clusters=4):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
    from sklearn.feature_selection import f_classif
    import torch
    import numpy as np

    if isinstance(emb_data, torch.Tensor):
        emb_data = emb_data.cpu().detach().numpy()

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans_labels = kmeans.fit_predict(emb_data)

    # Extract true cluster assignments
    true_labels_raw = [ground_true_gene_to_cluster_dict.get(gene, -1) for gene in genes_of_the_emb_data]

    # Convert labels to integers
    true_unique_labels = list(set(true_labels_raw))
    true_labels = [true_unique_labels.index(label) for label in true_labels_raw]

    kmeans_unique_labels = list(set(kmeans_labels))
    kmeans_labels_int = [kmeans_unique_labels.index(label) for label in kmeans_labels]

    # Compute evaluation metrics
    ari = adjusted_rand_score(true_labels, kmeans_labels_int)
    nmi = normalized_mutual_info_score(true_labels, kmeans_labels_int)
    fmi = fowlkes_mallows_score(true_labels, kmeans_labels_int)

    # Dimension importance
    # clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(emb_data, true_labels)
    # importances = clf.feature_importances_
    # Dimension importance based on correlation with true labels
    # Compute F-values and p-values using ANOVA
    F, p = f_classif(emb_data, true_labels)
    # print(f"emb_data.shape {emb_data.shape}, true_labels {len(true_labels)}, genes_of_the_emb_data {len(genes_of_the_emb_data)}")
    # print(true_labels)
    # print(genes_of_the_emb_data)
    # Normalize the F-values for interpretability
    #importances = F / sum(F)
    total_F = sum(F)
    importances = (F / total_F) if total_F != 0 else F

    # Printing the shapes of F and total_F
    #print(f"Shape of F: {F.shape}") # it printed out as Shape of F: (200,) or Shape of F: (256,)
    #print(f"Value of total_F: {total_F}") # total_F is a single float 

    return ari, nmi, fmi, importances



def compare_2_embeddings_with_ground_true_clusters(ori_ground_true_member_to_genes_dict, ori_token_emb1, ori_token_emb2, gene_to_idx1, gene_to_idx2, 
                       emb1_label="GA", emb2_label="GF", 
                       num_trials=500, num_of_clusters=4, random_seed=8, min_gene_num_for_a_group=4, shuffling=False):
    import pandas as pd
    import numpy as np
    if isinstance(ori_token_emb1, torch.Tensor):
        ori_token_emb1 = ori_token_emb1.detach().cpu().numpy()
    if isinstance(ori_token_emb2, torch.Tensor):
        ori_token_emb2 = ori_token_emb2.detach().cpu().numpy()
    random.seed(random_seed)
    clustering_metrics = []
    # Initialize 2D arrays for dimension importance accumulation
    # assuming both embeddings have the same dimension
    accumulated_importance_emb1 = np.zeros((num_trials, ori_token_emb1.shape[1]))
    accumulated_importance_emb2 = np.zeros((num_trials, ori_token_emb2.shape[1]))
    valid_trials = 0  # Counter to track the number of valid trials
    genes_in_both_models = get_overlapping_genes(gene_to_idx1.keys(), gene_to_idx2.keys())
    #ground_true_member_to_genes_dict = ori_ground_true_member_to_genes_dict
    # filter for the genes that exist in the models
    ori_ground_true_member_to_genes_in_models = {group: [gene for gene in genes if gene in genes_in_both_models] for group, genes in ori_ground_true_member_to_genes_dict.items()}
    # filter for the groups that have number of genes >= min_gene_num_for_a_group
    ground_true_member_to_genes_dict = {group: genes for group, genes in ori_ground_true_member_to_genes_in_models.items() if len(genes) >= min_gene_num_for_a_group}
    flatten_gene_list_in_ground_true_and_in_both_models = [gene for genes in ground_true_member_to_genes_dict.values() for gene in genes]
    print(f"After intersecting with both models, there are {len(ground_true_member_to_genes_dict)} clusters, {len(flatten_gene_list_in_ground_true_and_in_both_models)} genes!")
    if len(ground_true_member_to_genes_dict) <= num_of_clusters + 2:
        print(f"Aftering filtering for group length >= {min_gene_num_for_a_group}, number of groups {len(ground_true_member_to_genes_dict)}, too few, will skip!")
        return None, None
    gene_to_cluster_dict = {gene: group for group, gene_list in ground_true_member_to_genes_dict.items() for gene in gene_list}
    dimension_importance = {emb1_label: [], emb2_label: []}
    
    #print(f"ori {ori_token_emb1[:2, :5]}")
    for trial_idx in range(num_trials*3):
        if shuffling:
            # Make deep copies of the embedding matrices
            token_emb1 = copy.deepcopy(ori_token_emb1)
            token_emb2 = copy.deepcopy(ori_token_emb2)
            
            # Shuffle the rows of the copied embedding matrices
            np.random.shuffle(token_emb1)
            np.random.shuffle(token_emb2)
        else:
            token_emb1 = ori_token_emb1
            token_emb2 = ori_token_emb2
        #print(f"after {token_emb1[:2, :5]}")
        selected_keys = random.sample(list(ground_true_member_to_genes_dict.keys()), num_of_clusters)
        # print(f"selected_keys {selected_keys}")
        genes_of_interest = []
        for key in selected_keys:
            # print(f"key {key}: genes {ground_true_member_to_genes_dict[key]}")
            genes_of_interest += ground_true_member_to_genes_dict[key]
        unique_genes = [gene for gene in genes_of_interest if genes_of_interest.count(gene) == 1]
        
        genes_of_interest_and_in_both_models = get_overlapping_genes(genes_in_both_models, unique_genes)

        # only remain clusters with at least {min_gene_num_for_a_group} genes, and if there are < 2 clusters left, continue to next trial
        cluster_genecount_dict = Counter([gene_to_cluster_dict[gene] for gene in genes_of_interest_and_in_both_models])
        list_of_cluster_with_enough_genes = [cluster for cluster, gene_count in cluster_genecount_dict.items() if gene_count >= min_gene_num_for_a_group]
        if len(list_of_cluster_with_enough_genes) < 2:
            continue
        list_of_genes_in_good_clusters = [gene for gene in genes_of_interest_and_in_both_models if gene_to_cluster_dict[gene] in list_of_cluster_with_enough_genes]


        idx_of_selected_genes1 = [gene_to_idx1[gene] for gene in list_of_genes_in_good_clusters]
        emb_selected_genes1 = token_emb1[idx_of_selected_genes1]
        
        idx_of_selected_genes2 = [gene_to_idx2[gene] for gene in list_of_genes_in_good_clusters]
        emb_selected_genes2 = token_emb2[idx_of_selected_genes2]

        emb1_ari, emb1_nmi, emb1_fmi, emb1_importance = emb_to_cluster_performance_metrics(emb_selected_genes1, list_of_genes_in_good_clusters, gene_to_cluster_dict)
        emb2_ari, emb2_nmi, emb2_fmi, emb2_importance = emb_to_cluster_performance_metrics(emb_selected_genes2, list_of_genes_in_good_clusters, gene_to_cluster_dict)
        
        # print(emb1_importance.shape)
        # print(emb2_importance.shape)
        # print(accumulated_importance_emb1.shape)
        # print(accumulated_importance_emb2.shape)

        accumulated_importance_emb1[valid_trials] = emb1_importance
        accumulated_importance_emb2[valid_trials] = emb2_importance
        
        valid_trials += 1
                # appending the metrics for both embeddings
        clustering_metrics.append([emb1_label, "ARI", emb1_ari])
        clustering_metrics.append([emb1_label, "NMI", emb1_nmi])
        clustering_metrics.append([emb1_label, "FMI", emb1_fmi])

        clustering_metrics.append([emb2_label, "ARI", emb2_ari])
        clustering_metrics.append([emb2_label, "NMI", emb2_nmi])
        clustering_metrics.append([emb2_label, "FMI", emb2_fmi])
        if valid_trials == num_trials:
            break

    # convert the list to a DataFrame
    clustering_metrics_df = pd.DataFrame(clustering_metrics, columns=["Embedding_label", "Metric", "Value"])
    # Calculate the mean importance for each dimension across valid trials using vector operations
    mean_importance_emb1 = accumulated_importance_emb1[:valid_trials].mean(axis=0)
    mean_importance_emb2 = accumulated_importance_emb2[:valid_trials].mean(axis=0)

    mean_importance = {
        emb1_label: mean_importance_emb1.tolist(),
        emb2_label: mean_importance_emb2.tolist()
    }
    
    return clustering_metrics_df, mean_importance




def wrapper_cmp_two_embeddings_with_group_true_clustering(members_dict, token_emb1, token_emb2, gene_to_idx1, gene_to_idx2,
                                     emb1_label, emb2_label, save_path, file_prefix,
                                     num_of_clusters=4, num_trials=500, show_image=False, random_seed=8, min_gene_num_for_a_group=4):
    """
    Analyze and visualize embeddings by comparing them with ground-truth clusters.
    This function generates metrics, creates a null distribution, combines results,
    and then visualizes and saves the results.

    Parameters:
        - members_dict: Ground truth clusters as a dictionary.
        - token_emb1, token_emb2: Embeddings to compare.
        - gene_to_idx1, gene_to_idx2: Gene to index mappings for the embeddings.
        - emb1_label, emb2_label: Labels for the embeddings.
        - save_path: Directory to save the results.
        - file_prefix: Prefix for the saved files.
        - num_of_clusters: Number of clusters for analysis (default is 4).
        - num_trials: Number of trials for generating null distribution (default is 500).
        - show_image: Whether to display the generated plots (default is False).
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Calculate clustering metrics
    clustering_metrics_df, dimension_importance_dict = compare_2_embeddings_with_ground_true_clusters(
        members_dict,
        token_emb1,
        token_emb2,
        gene_to_idx1,
        gene_to_idx2,
        emb1_label=emb1_label,
        emb2_label=emb2_label,
        num_of_clusters=num_of_clusters,
        num_trials=num_trials,
        random_seed=random_seed,
        min_gene_num_for_a_group=min_gene_num_for_a_group
    )
    
    # Generate null distribution generate_null_distribution_for_metrics
    null_distribution_df, null_dimension_importance_dict = compare_2_embeddings_with_ground_true_clusters(
        members_dict,
        token_emb1,
        token_emb2,
        gene_to_idx1,
        gene_to_idx2,
        emb1_label="Null_" + emb1_label,
        emb2_label="Null_" + emb2_label,
        num_of_clusters=num_of_clusters,
        num_trials=num_trials,
        random_seed=random_seed,
        min_gene_num_for_a_group=min_gene_num_for_a_group,
        shuffling=True
    )
    
    if dimension_importance_dict != None:
        save_dimension_importance(dimension_importance_dict, save_path, file_prefix=f'{file_prefix}real.')
    if null_dimension_importance_dict != None:
        save_dimension_importance(null_dimension_importance_dict, save_path, file_prefix=f'{file_prefix}null.')
    
    # Concatenate the original metrics with the null distribution
    if clustering_metrics_df is not None and not clustering_metrics_df.empty and null_distribution_df is not None and not null_distribution_df.empty:
        combined_metrics_df = pd.concat([clustering_metrics_df, null_distribution_df], ignore_index=True)
        # Visualize and save results
        visualize_and_compare_metrics(
            combined_metrics_df,
            emb1_label,
            emb2_label,
            save_path=save_path,
            file_prefix=file_prefix,
            show_image=show_image
        )
        visualize_and_compare_metrics(
            combined_metrics_df,
            emb1_label,
            "Null_" + emb1_label,
            save_path=save_path,
            file_prefix=file_prefix + "Null_vs_" + emb1_label + ".",
            show_image=show_image
        )
        visualize_and_compare_metrics(
            combined_metrics_df,
            emb2_label,
            "Null_" + emb2_label,
            save_path=save_path,
            file_prefix=file_prefix + "Null_vs_" + emb2_label + ".",
            show_image=show_image
        )

        save_clustering_metrics_and_statistics(
            combined_metrics_df,
            save_path=save_path,
            file_prefix=file_prefix
        )

def save_clustering_metrics_and_statistics(clustering_metrics_df, save_path, file_prefix=''):
    # Create the save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save the original clustering_metrics_df to CSV file
    original_file_name = os.path.join(save_path, f"{file_prefix}clustering_metrics.csv")
    clustering_metrics_df.to_csv(original_file_name, index=False)
    #print(f"Original clustering metrics saved to {original_file_name}")
    
    # Calculate descriptive statistics for each group
    statistics_df = clustering_metrics_df.groupby(['Embedding_label', 'Metric'])['Value'].describe().reset_index()
    
    # Save the descriptive statistics to another CSV file
    statistics_file_name = os.path.join(save_path, f"{file_prefix}clustering_stat.csv")
    statistics_df.to_csv(statistics_file_name, index=False)
    #print(f"Descriptive statistics saved to {statistics_file_name}")

def save_dimension_importance(mean_importance_dict, save_path, file_prefix=''):
    if mean_importance_dict == None:
        return
    # Create the save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for emb_label, mean_importance in mean_importance_dict.items():

        # Convert mean_importance to a DataFrame where each key-value becomes a column
        dimension_importance_df = pd.DataFrame({emb_label: mean_importance})

        # Save the data to a CSV file
        original_file_name = os.path.join(save_path, f"{file_prefix}{emb_label}.fi.csv")
        dimension_importance_df.to_csv(original_file_name, index=False)

        #print(f"dimension_importance_df saved to {original_file_name}")



def visualize_and_compare_metrics(clustering_metrics_df, emb1_label, emb2_label, save_path=None, file_prefix='', fig_height=5, show_image=True):
    # print(type(clustering_metrics_df))
    # print(clustering_metrics_df.head())
    # print(clustering_metrics_df['Embedding_label'] == emb1_label)
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    # Create the save directory if it does not exist
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    g = sns.catplot(x="Metric", y="Value", hue="Embedding_label", kind="box", data=clustering_metrics_df, height=fig_height, aspect=1.5)
    plt.title('Box Plots for Clustering Metrics')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    g._legend.remove()  # Remove the original legend
    g.add_legend(loc='upper right')  # Add a new legend at the desired location
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    # if save_path is not None:
    #     plt.savefig(os.path.join(save_path, f"{file_prefix}box_plot_with_null_distr.png"))
    if show_image:
        plt.show()
    plt.close(g.fig)

    # Mann-Whitney U test and Wilcoxon Signed-Rank Test
    metrics = ["ARI", "NMI", "FMI"]
    results = []
    for metric in metrics:
        emb1_group = clustering_metrics_df.loc[
            (clustering_metrics_df['Embedding_label'] == emb1_label) & 
            (clustering_metrics_df['Metric'] == metric), 'Value']
        
        emb2_group = clustering_metrics_df.loc[
            (clustering_metrics_df['Embedding_label'] == emb2_label) & 
            (clustering_metrics_df['Metric'] == metric), 'Value']
        
        # Calculate descriptive statistics for each group
        emb1_mean = emb1_group.mean()
        emb1_std = emb1_group.std()
        emb2_mean = emb2_group.mean()
        emb2_std = emb2_group.std()
        
        # Perform Mann-Whitney U test
        mannwhitney_stat, mannwhitney_p = mannwhitneyu(emb1_group, emb2_group)
        
        # Perform Wilcoxon Signed-Rank Test
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(emb1_group, emb2_group)
        except ValueError as e:
            wilcoxon_stat, wilcoxon_p = None, None
            print(f"Wilcoxon Signed-Rank Test not applicable: {e}")
        
        # Append results
        results.append({
            'Metric': metric,
            'mannwhitney_stat': mannwhitney_stat,
            'mannwhitney_p': mannwhitney_p,
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'Comparison': f"{emb1_label} vs {emb2_label}",
            f'{emb1_label}_mean': emb1_mean,
            f'{emb1_label}_std': emb1_std,
            f'{emb2_label}_mean': emb2_mean,
            f'{emb2_label}_std': emb2_std
        })
        # Print results
        if show_image:
            print(f"Comparing {emb1_label} {metric} vs {emb2_label} {metric}:")
            print(f"Mann-Whitney U: Statistic: {mannwhitney_stat}, P-value: {mannwhitney_p}")
            print(f"Wilcoxon Signed-Rank: Statistic: {wilcoxon_stat}, P-value: {wilcoxon_p}")
            print(f"{emb1_label} Mean: {emb1_mean}, Std: {emb1_std}")
            print(f"{emb2_label} Mean: {emb2_mean}, Std: {emb2_std}\n")

    # Save results to CSV
    if save_path is not None:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_path, f"{file_prefix}stat_results.csv"), index=False)

def print_uniq_value_of_each_column(all_stat_df, max_unique_value_num=40):
    for column in all_stat_df.columns:
        try:
            unique_values = all_stat_df[column].unique()
        except TypeError:
            # Convert lists to tuples for hashability
            unique_values = all_stat_df[column].apply(lambda x: tuple(x) if isinstance(x, list) else x).unique()
    
        if len(unique_values) <= max_unique_value_num and len(unique_values) > 1:
            # Convert all unique values to strings, add quotes, and join them with a comma
            unique_values_str = ', '.join(f"'{value}'" if isinstance(value, str) else str(value) for value in unique_values)
            print(f"Column: {column}")
            print(f"Unique Values: [{unique_values_str}]")
            print()

def get_custom_dataset_label_geneset_dict():
    config = get_config()
    custom_geneset_files = glob.glob(config.project_path  + "/data/external/custom_genesets/*.tsv"
                                    )
    custom_dataset_label_geneset_dict = {}
    for custom_geneset_file in  custom_geneset_files:
        basename = os.path.basename(custom_geneset_file)
        dataset_label = basename.replace(".tsv", "")
        diseases_dict = {}
        
        # Open the text file and read the lines
        with open(custom_geneset_file, 'r') as file:
            for line in file:
                # Strip newline characters and other trailing whitespace, then split by space
                if "\t" in line:
                    parts = line.strip().split("\t")
                    key = parts[0]
                    # only one tab, which is between geneset_label and genes
                    if " " in parts[1]:
                        values = parts[1].split()
                    # genes are separated by tabs too
                    else:
                        values = parts[1:]
                # all are separated by spaces
                else:
                    parts = line.strip().split()
                    # The first part is the key, the rest are the values
                    key = parts[0]
                    values = parts[1:]
                # Add to the dictionary
                diseases_dict[key] = values
        custom_dataset_label_geneset_dict[dataset_label] = diseases_dict
    return custom_dataset_label_geneset_dict


def get_geneset_labels_for_clf_filt():
    config = get_config()
    geneset_labels_for_clf_filt_file = config.project_path + "/data/external/Enrichr/092023/geneset_labels_for_clf_filt.txt"
    # List to store the column data
    geneset_labels_for_clf_filt = []
    
    # Open the TSV file
    with open(geneset_labels_for_clf_filt_file, 'r', newline='') as tsvfile:
        # Create a CSV reader specifying the delimiter as a tab
        reader = csv.reader(tsvfile, delimiter='\t')
        # Read each row in the file
        for row in reader:
            # Since it's a single-column file, take the first element
            geneset_labels_for_clf_filt.append(row[0])
    return geneset_labels_for_clf_filt

def get_ens_and_symbol_2_ga_genes_and_type(file_path=f"{config.proj_path}/data/external/ARCHS/human_gene_v2.2.h5.genes_meta.tsv"):
    """
    This function return:
    1. a dict of Ensembl ID (col2) or gene symbol (col3) to gene symbol used by GA, based on the meta data of genes in ARCHS4 dataset.
    2. a dict of Ensembl ID (col2) or gene symbol (col3) to gene type, like protein coding or lncRNA
    """
    ens_and_symbol_2_ga_genes = {}
    gene2type = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            gene_in_ga = row[2]
            ens_and_symbol_2_ga_genes[row[1]] = gene_in_ga
            ens_and_symbol_2_ga_genes[gene_in_ga] = gene_in_ga
            gene2type[row[1]] = row[0]
            gene2type[gene_in_ga] = row[0]
    return ens_and_symbol_2_ga_genes, gene2type
