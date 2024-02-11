import os
os.environ['SPECIES'] = "Mouse"
import pickle
from utils.utils import get_config
m_config = get_config()

def read_homolog_file(file_path):
    """
    Read a file containing homologous genes and return a dictionary with human genes as keys
    and a list of corresponding mouse homolog genes as values. It accounts for multiple DB Class Keys
    for a single human gene.

    Parameters:
    file_path (str): The path to the text file containing the homologous gene data.

    Returns:
    dict: A dictionary with human genes as keys and lists of mouse genes as values.
    """
    # Initialize empty dictionaries to store the human and mouse genes with their DB Class Key
    human_genes = {}
    mouse_genes = {}
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Split each line into components
            db_class_key, organism, taxon_id, symbol = line.strip().split('\t')[:4]
            
            # Populate the human and mouse dictionaries with their symbols and DB Class Key
            if taxon_id == '9606':  # Human genes have taxon ID 9606
                # Account for multiple DB Class Keys for a single human gene
                if symbol not in human_genes:
                    human_genes[symbol] = [db_class_key]
                else:
                    if db_class_key not in human_genes[symbol]:
                        human_genes[symbol].append(db_class_key)
            elif taxon_id == '10090':  # Mouse genes have taxon ID 10090
                mouse_genes[symbol] = db_class_key

    # Initialize a dictionary to hold the final human to mouse homolog mappings
    homolog_dict = {}

    # Cross-checking human and mouse genes to ensure they share the same DB Class Key
    for human_gene, human_db_keys in human_genes.items():
        # Find mouse genes with the matching DB Class Key
        matching_mouse_genes = [mouse_gene for mouse_gene, mouse_db_key in mouse_genes.items() if mouse_db_key in human_db_keys]
        
        # If there are matching mouse genes, add them to the homolog dictionary
        if matching_mouse_genes:
            # Ensure unique list of matching genes
            unique_mouse_genes = list(set(matching_mouse_genes))
            homolog_dict[human_gene] = unique_mouse_genes

    return homolog_dict

def reverse_homolog_mapping(homolog_dict):
    """
    Create a dictionary mapping mouse genes to lists of corresponding human homolog genes
    based on an existing dictionary mapping human genes to mouse homologs.

    Parameters:
    homolog_dict (dict): A dictionary with human genes as keys and lists of mouse genes as values.

    Returns:
    dict: A dictionary with mouse genes as keys and lists of human genes as values.
    """
    mouse_to_human_dict = {}

    # Iterate through each human gene and its corresponding mouse genes
    for human_gene, mouse_genes in homolog_dict.items():
        for mouse_gene in mouse_genes:
            # If the mouse gene isn't a key in the dictionary yet, add it
            if mouse_gene not in mouse_to_human_dict:
                mouse_to_human_dict[mouse_gene] = [human_gene]
            else:
                # Append the human gene to the existing list if it's not already included
                if human_gene not in mouse_to_human_dict[mouse_gene]:
                    mouse_to_human_dict[mouse_gene].append(human_gene)

    return mouse_to_human_dict

def get_homolo_mg():
    """
    return h2m_genes, m2h_genes, h2m_one2one, m2h_one2one
    """
    mg_homolog_file = m_config.proj_path + "/data/external/Homolo/MouseGenomeInfo/HOM_MouseHumanSequence.rpt"
    pickle_file = m_config.proj_path + "/data/external/Homolo/MouseGenomeInfo/HOM_MouseHumanSequence.rpt.homolog.pickle"
    
    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        # Read the data from the pickle file
        with open(pickle_file, 'rb') as f:
            h2m_genes, m2h_genes, h2m_one2one, m2h_one2one = pickle.load(f)
    else:

        h2m_genes = read_homolog_file(mg_homolog_file)
        # Get the reverse mapping from mouse to human
        m2h_genes = reverse_homolog_mapping(h2m_genes)

        # one2one human genes are the key, the keys are good, the value may be not unique
        one2one_human_genes = {h:ms[0] for h, ms in h2m_genes.items() if len(ms)==1}

        # one2one mouse genes are the key, the keys are good, the value may be not unique
        one2one_mouse_genes = {m:hs[0] for m, hs in m2h_genes.items() if len(hs)==1}

        h2m_one2one = {h:m for h,m in one2one_human_genes.items() if m in one2one_mouse_genes}
        m2h_one2one = {m:h for m,h in one2one_mouse_genes.items() if h in one2one_human_genes}
        with open(pickle_file, 'wb') as f:
            pickle.dump((h2m_genes, m2h_genes, h2m_one2one, m2h_one2one), f)

    return h2m_genes, m2h_genes, h2m_one2one, m2h_one2one
