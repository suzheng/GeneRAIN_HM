import os
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
from utils.utils import get_device, get_model, get_config
from train.common_params_funs import get_gene2idx
from utils.params import params as params_class
from eval.eval_utils import get_self_emb
from eval_m.m_eval_utils import eval_MUSE_emb
from eval_m.m_visual_utils import plot_PCA, plot_histogram
from data_m.get_homolo_mg import get_homolo_mg
from data_m.m_data_utils import split_human_mouse_emb_and_genes, split_dict, dict_to_file, write_string_to_file, save_matrix_and_ids, load_matrix_and_ids
from data_m.m_data_utils import read_gz_tsv, print_unique_values, df_cols_to_dict, query_genes, get_combined_dna_seq_conserv_df, add_anno_info
from dna.dna_utils import file_to_dict
import numpy as np
from data.data_utils import output_emb
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
import glob
import pandas as pd
import pickle
import os
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
from utils.utils import get_device, get_model, get_config
from train.common_params_funs import get_gene2idx
from utils.params import params
from eval.eval_utils import get_self_emb
from sklearn.metrics.pairwise import cosine_similarity
from eval_m.m_eval_utils import get_n_closest_tokens, reorder_symmetric_matrix, mean_and_std_of_arrays, find_rank_of_similarity, get_similarity, read_tsv_to_dict
from eval_m.m_visual_utils import density_colored_scatter, plot_PCA, plot_histogram, plot_comparison_box, plot_custom_ecdf, plot_comparison_bar_with_ci
from data_m.get_homolo_mg import get_homolo_mg
from data_m.m_data_utils import split_human_mouse_emb_and_genes, split_dict, dict_to_file, write_string_to_file, save_matrix_and_ids, load_matrix_and_ids, list_of_dicts_to_dataframe
from data_m.m_data_utils import read_gz_tsv, print_unique_values, df_cols_to_dict, query_genes, get_combined_dna_seq_conserv_df, add_anno_info, get_gene2type_dict
from dna.dna_utils import file_to_dict
import numpy as np
from data.data_utils import output_emb
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from pySankey.sankey import sankey  # Assuming you've installed pySankeya
from collections import Counter
import os
import pandas as pd
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
import seaborn as sns
from utils.utils import get_config
from eval_m.m_eval_utils import read_rank_dict, get_similarity
from eval_m.m_visual_utils import density_colored_scatter, plot_side_by_side_histogram, plot_lines
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
from data_m.m_data_utils import split_human_mouse_emb_and_genes, split_dict, dict_to_file, write_string_to_file, save_matrix_and_ids, load_matrix_and_ids, list_of_dicts_to_dataframe
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from data_m.get_homolo_mg import get_homolo_mg
from utils.params import params as params_class
import pickle
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
import os
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
from utils.utils import get_device, get_model, get_config
from train.common_params_funs import get_gene2idx
from utils.params import params
from eval.eval_utils import get_self_emb
from sklearn.metrics.pairwise import cosine_similarity
from eval_m.m_eval_utils import get_n_closest_tokens, reorder_symmetric_matrix, mean_and_std_of_arrays, find_rank_of_similarity, get_similarity, n_closest_tokens_to_gene_pair_list
from eval_m.m_visual_utils import density_colored_scatter, plot_PCA, plot_histogram, plot_comparison_box, plot_custom_ecdf, plot_comparison_bar_with_ci, plot_side_by_side_histogram
from data_m.get_homolo_mg import get_homolo_mg
from data_m.m_data_utils import split_human_mouse_emb_and_genes, split_dict, dict_to_file, write_string_to_file, save_matrix_and_ids, load_matrix_and_ids
from data_m.m_data_utils import read_gz_tsv, print_unique_values, df_cols_to_dict, query_genes, get_combined_dna_seq_conserv_df, add_anno_info
from dna.dna_utils import file_to_dict
import numpy as np
from data.data_utils import output_emb
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import textwrap
import matplotlib.gridspec as gridspec
from scipy import stats
from eval.visual_utils import pval2stars
import re
import statsmodels.api as sm
from scipy import stats
import pandas as pd
from itertools import chain
from collections import Counter
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
# import glob
# from data_m.get_homolo_mg import get_homolo_mg
import os
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
from utils.utils import get_device, get_model, get_config
from utils.params import params
import glob
import csv
import re
import numpy as np
import pandas as pd
import pickle
from eval.eval_utils import print_uniq_value_of_each_column
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from eval_m.m_visual_utils import plot_PCA, plot_histogram
from eval_m.m_eval_utils import eval_specified_emb
import random
import os
import glob
from eval.visual_utils import Naming_Json
# Additional imports for data and model handling
import numpy as np
from train.common_params_funs import get_gene2idx
from utils.utils import get_device, get_model, get_config, get_gene2vec
from eval.eval_utils import get_self_emb
from utils.params import params as params_class
from data_m.get_homolo_mg import get_homolo_mg
from data_m.m_data_utils import write_string_to_file, split_human_mouse_emb_and_genes, split_dict, dict_to_file
from data.data_utils import output_emb
import os
from eval.visual_utils import Naming_Json
from utils.utils import get_config
import pickle
import os
from eval.visual_utils import (get_naming_dict, filter_dataframe, FigureStyle, pval2stars, Naming_Json, export_to_excel)
from utils.utils import get_device, get_model, get_config
from train.common_params_funs import get_gene2idx
from utils.params import params
from eval.eval_utils import get_self_emb
from sklearn.metrics.pairwise import cosine_similarity
from eval_m.m_eval_utils import get_n_closest_tokens, reorder_symmetric_matrix, mean_and_std_of_arrays, find_rank_of_similarity, eval_simi_mat
from eval_m.m_visual_utils import plot_PCA, plot_histogram
from data_m.get_homolo_mg import get_homolo_mg
from data_m.m_data_utils import split_human_mouse_emb_and_genes, split_dict, dict_to_file, write_string_to_file, save_matrix_and_ids, load_matrix_and_ids
import numpy as np
from data.data_utils import output_emb
from eval_m.muse_utils import load_muse_result, get_gene_to_ori_case_dict, get_muse_result_uid_dir
import os
import pickle
