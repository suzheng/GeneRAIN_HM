# GeneRAIN Human-Mouse

## About GeneRAIN Human-Mouse

This GitHub repository presents our study on gene expression in humans and mice, utilizing deep learning on 410K human and 366K mouse bulk RNA-seq samples to investigate gene function and disease associations across species. Our research leverages our Transformer-based models [GeneRAIN] (https://github.com/suzheng/GeneRAIN) and cross-species gene embedding alignment to analyze RNA-level similarities between human and mouse genes, offering new insights into their evolutionary and functional relationships. This project enhances our understanding of mouse genes in biomedical research, providing a novel methodology for cross-species omics analysis.

## Repository Contents

- **Model Training Framework**: Scripts and guidelines for training the GeneRAIN models on the prepared datasets.
- **Statistical analyses**: Notebooks to perform statistical analysis and generate the figures and tables in the paper. 


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
	- Download the data from figshare. Refer to the data [README file](data/README.md) in the data directory for descriptions and details of the files.
	- Extract the downloaded `tar.gz` file.
	- Download the ARCHS4 [`human_gene_v2.2.h5`](https://maayanlab.cloud/archs4/download.html) and [`mouse_gene_v2.2.h5`](https://maayanlab.cloud/archs4/download.html) files, and move them to folder `data/external/ARCHS/`
	- Perform data preprocessing and normalization following the steps in [GeneRAIN] (https://github.com/suzheng/GeneRAIN)


## Train the Models

Once you have set up everything, you are ready to begin training the models. The training process is managed through the script [`src/train/pretrain.py`](src/train/pretrain.py), which accepts three parameters:

- `--epoch_from`: Specifies the starting epoch number of training, beginning from 1. (Type: int, Default: None)
- `--epoch_to`: Specifies the ending epoch number of training. (Type: int, Default: None)
- `--exp_label`: Provides an experiment label for the output. (Type: str, Default: None)

### Configuration via JSON File

- All model and training hyperparameters are specified in a JSON file. Please find files in folder [`jsons`](jsons) and the information in the table below for all the json files used for different models. 
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


## Statistical analyses

The following Jupyter notebooks detail the statistical analyses performed in our study, from data setup and embedding alignment to phenotype associations and model evaluations.

- **[`prepare_data_scripts_for_MUSE.ipynb`](notebooks/anal/prepare_data_scripts_for_MUSE.ipynb):** Sets up data and scripts for aligning embeddings with MUSE.
- **[`process_shared_emb_results.ipynb`](notebooks/anal/process_shared_emb_results.ipynb):** Generates similarity matrices from shared embedding alignment experiments, arranges data for MUSE supervised alignment, and produces a PCA plot for one of the experiments.
- **[`anal_emb_from_MUSE.ipynb`](notebooks/anal/anal_emb_from_MUSE.ipynb):** Calculates ranks and similarities from MUSE embeddings and saves the findings in a pickle file.
- **[`cmp_align_exps.ipynb`](notebooks/anal/cmp_align_exps.ipynb):** Compares results from various embedding alignment experiments.
- **[`cal_LECIF_scores.ipynb`](notebooks/anal/cal_LECIF_scores.ipynb):** Calculate the LECIF scores for genes.
- **[`eval_simi_assoc.ipynb`](notebooks/anal/eval_simi_assoc.ipynb):** Analyzes associations between phenotypes, DNA features, and gene embeddings.
- **[`anal_lncRNA_pseudo.ipynb`](notebooks/anal/anal_lncRNA_pseudo.ipynb):** Examines results from the protein-coding+lncRNAs+pseudogenes model.
- **[`plot_loss_vs_epochs.ipynb`](notebooks/anal/plot_loss_vs_epochs.ipynb):** Plots training loss against epochs to visualize model performance.






