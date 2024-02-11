import os

from utils.utils import get_config
os.environ['SPECIES'] = "Mouse"
config = get_config()
# Function to read a dual-column file into a dictionary
def file_to_dict(filename):
    with open(filename, 'r') as file:
        d = {}
        for line in file:
            key, value = line.strip().split()  # Adjust splitting method if needed
            d[key] = value
    return d

def swap_key_values(d):
    new_dict = {}
    for key, value in d.items():
        # If the value is already a key in the new dictionary, append the old key to the list of values
        if value in new_dict:
            new_dict[value].append(key)
        else:
            # Otherwise, create a new list with the single old key
            new_dict[value] = [key]
    return new_dict

def get_transcript2gene():
    human_file = config.proj_path + "/data/dna/gtf/Homo_sapiens.GRCh38.110.gtf.gz.transcript2gene"
    mouse_file = config.proj_path + "/data/dna/gtf/Mus_musculus.GRCm39.110.gtf.gz.transcript2gene"
    # Read each file into a dictionary
    dict1 = file_to_dict(human_file)
    dict2 = file_to_dict(mouse_file)

    dict2 = {k: f"m_{v}" for k, v in dict2.items()}
    # Combine the dictionaries
    transcript2gene = {**dict1, **dict2}
    gene2transcripts = swap_key_values(transcript2gene)
    return transcript2gene, gene2transcripts