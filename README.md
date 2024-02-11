# GeneRAIN Human-Mouse

## About GeneRAIN Human-Mouse

GeneRAIN models, based on BERT and GPT Transformer architectures, are trained on an extensive dataset of 410K human bulk RNA-seq samples. These models focus on analyzing gene network correlations and developing robust gene representations. By leveraging bulk RNA-seq data, GeneRAIN distinguishes itself from conventional models that primarily use single-cell RNA-seq data. The combination of varied model architectures with the 'Binning-By-Gene' normalization method allows GeneRAIN to effectively decode a broad range of biological information. This repository serves as a platform providing the necessary code and instruction for dataset preparation, training of the GeneRAIN models, and their application to new samples, utilizing this specialized normalization technique.

## Repository Contents
- **Data Preparation Scripts**: Tools to prepare and preprocess the dataset for training the GeneRAIN models.
- **Model Training Framework**: Scripts and guidelines for training the GeneRAIN models on the prepared datasets.
- **Normalization Tools**: Implementation of the 'Binning-By-Gene' normalization method for processing new expression data and preparing it for model input.
- **Utilization of Model Checkpoints**: Using pre-trained models and checkpoints for applying GeneRAIN to new datasets.


## Installation and Setup

To successfully install and set up this project, ensure that you have a Linux environment equipped with CUDA-capable GPUs. Additionally, the corresponding NVIDIA drivers and CUDA toolkit must be properly installed to fully leverage the GPU acceleration capabilities required by the project.

1. **Clone the Repository**:

	```bash
	git clone https://github.com/suzheng/GeneRAIN_HM.git
	```
2. **Set Up a Virtual Environment**:
Before installing the package, it's recommended to set up a virtual environment. This will keep the project's dependencies isolated from your global Python environment.

	```
	# Navigate to the project directory
	cd [project-directory]
	
	# Create the folders for training output
	mkdir -p results/eval_data results/logs  results/models results/debugging/eval_data results/debugging/logs  results/debugging/models
	
	# Create a virtual environment named 'generain'
	python -m venv generain_hm
	
	# Activate the virtual environment
	source generain_hm/bin/activate
	```

3. **Install Dependencies**:
Once the virtual environment is activated, install the project's dependencies.

	```
	pip install -r requirements.txt
	```

4. **Install the Package**:
Install the project package within the virtual environment. 

	```
	pip install .
	# For development, use command below instead:
	pip install -e .
	```
After installation, the package and its modules can be imported into other Python scripts or notebooks.


5. **Prepare the Data**:
	- Download the data from Zenodo. Refer to the data [README file](data/README.md) in the data directory for descriptions and details of the files.
	- Extract the downloaded `tar.gz` file.
	- Move the downloaded `human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_*.npy` files to `data/external/ARCHS/normalize_each_gene/` in the extracted folder.
	- Download the ARCHS4 [`human_gene_v2.2.h5`](https://maayanlab.cloud/archs4/download.html) file, and move the `human_gene_v2.2.h5` file to folder `data/external/ARCHS/`


## Train the Models

Once you have set up everything, you are ready to begin training the models. The training process is managed through the script [`src/train/pretrain.py`](src/train/pretrain.py), which accepts three parameters:

- `--epoch_from`: Specifies the starting epoch number of training, beginning from 1. (Type: int, Default: None)
- `--epoch_to`: Specifies the ending epoch number of training. (Type: int, Default: None)
- `--exp_label`: Provides an experiment label for the output. (Type: str, Default: None)
- Please note that, the dataset in the Zenodo repo was normalized by 'Binning-By-Gene' method, and it is only for training `GPT_Binning_By_Gene`, `BERT_Pred_Expr_Binning_By_Gene` and `BERT_Pred_Genes_Binning_By_Gene` models.

### Configuration via JSON File

- All model and training hyperparameters are specified in a JSON file. Please find folders [`jsons`](jsons) for all the json files used for different GeneRAIN models. 
- The filename of this JSON configuration should be set in the environmental variable `PARAM_JSON_FILE`.

| Experiment            | Experiment ID                              |
|---------------------------|-----------------------------------------------|
| `human-mouse bert`             | `exp16_hm_BERT_coding`                        |
| `human-mouse gpt`              | `exp6_hm_GPT_coding`                          |
| `human-mouse bert+lncrna+pseudogenes`      | `exp20_hm_BERT_coding_lncRNA_pseudo`          |
| `human-mouse gpt+lncrna+pseudogenes`       | `exp10_hm_GPT_coding_lncRNA_pseudo`           |
| `human-mouse gpt shared emb 1`          | `exp30_GPT_hm_merge0_coding`                  |
| `human-mouse gpt shared emb 2`          | `exp31_GPT_hm_merge1_coding`                  |
| `human-mouse gpt shared emb 3`          | `exp32_GPT_hm_merge2_coding`                  |
| `human + pseudo non-human species gpt`          | `exp41_GPT_hm_fakenh_coding`                  |
| `mouse gpt`           | `exp1_mouse_GPT_coding`                       |
| `human gpt`           | `exp71_human_GPT_coding`                      |

### Debugging

- To run the training in debug mode, set the environment variable `RUNNING_MODE` to `debug`.

### Parallelism with DDP

- The training script utilizes Distributed Data Parallel (DDP) for parallelism.

### Example Scripts
- Example PBS scripts for submitting the training job can be found at [`src/examples`](src/examples).

### Output and Logging

- After training, the model checkpoints will be saved to the directory `results/models/pretrain/`.
- Tensorboard logs of the training loss are saved in `results/logs/pretrain/`. These logs can be viewed with the command `tensorboard --logdir=LOGS_FOLDER_OF_THE_EXPERIMENT`.
- For detailed usage of Tensorboard, please refer to its [official website](https://www.tensorflow.org/tensorboard).

Ensure that you have configured all necessary parameters and environment variables before initiating the training process.





