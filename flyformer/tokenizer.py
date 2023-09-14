"""
Flyformer tokenizer.

Input data:
Required format: 1e4-normalized and log1p-transformer scRNAseq data (.loom)
See flyformer.data_preprocessing for necessary steps
Required row (gene) attribute: "var_names": gene name (Flybase)
Optional col (cell) attributes: any other cell metadata can be passed on to the
tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from flyformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", 
                               "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

import argparse
from datasets import Dataset
import loompy as lp
import logging
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

logger = logging.getLogger(__name__)

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"



class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict: dict = None,
        nproc: int = 1,
        gene_median_file : Path = GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_percentile_file : Path
            Path to pickle file containing dictionary of percentile cutoffs of
            gene expression based on whole flyformer-corpus. The token
            dictionary will be created based on this object.
            { <str>: ( <int>|<str>, <float> ) ... } = 
            { gene: ( 20: <20th perc>, ... 80: <80th perc> ) }
        """
        # dictionary of custom attributes 
        # {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene expression percentiles (cutoffs)
        with open(gene_median_file, "rb") as f:
            self.gene_percentiles = pickle.load(f)

        self.gene_tokens = {}
        for gene, percentiles in self.gene_percentiles:
            self.gene_tokens = { 
                **self.gene_tokens, 
                **{ f"{gene}_{perc[0]}" for perc in percentiles }
            }

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for 
        # selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    @staticmethod
    def tokenize_cell(gene_vector, gene_tokens):
        """
        Convert normalized gene expression vector to tokenized rank value 
        encoding.
        """
        # create array of gene vector with token indices
        # mask undetected genes
        nonzero_mask = np.nonzero(gene_vector)[0]
        # sort by median-scaled gene values
        sorted_indices = np.argsort(-gene_vector[nonzero_mask])
        # tokenize
        sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
        return sentence_tokens

    def tokenize_file(self, loom_file_path):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists is True:
                filter_pass_loc = np.where(
                    [True if i == 1 else False for i in data.ca["filter_pass"]]
                )[0]
            elif var_exists is False:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / subview.ca.n_counts
                    * 10_000
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def tokenize_files(self, loom_data_directory):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            loom_cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        file_found = 0
        for loom_file_path in loom_data_directory.glob("*.loom"):
            file_found = 1
            print(f"Tokenizing {loom_file_path}")
            file_tokenized_cells, file_cell_metadata = self.tokenize_file(
                loom_file_path
            )
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in loom_cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f"No .loom files found in directory {loom_data_directory}.")
            raise
        return tokenized_cells, cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata):
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        def truncate(example):
            example["input_ids"] = example["input_ids"][0:2048]
            return example

        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length


def parse_args():
    parser = argparse.ArgumentParser(
            "Tokenize cell data based on expression values")
    parser.add_argument("--tdigests", "-t", required=True, 
                        type=argparse.FileType('r'))
    parser.add_argument("--loom", "-l", required=True 
                        type=argparse.FileType('r'))
    parser.add_argument("--token_dict", "-t", required=True 
                        type=argparse.FileType('r'))
    parser.add_argument("--output", "-o", required=True 
                        type=argparse.FileType('w'))
    return parser.parse_args()


def main():
    loom = sys.argv[1]

    with open(args.tdigests, "rb") as fp:
        tdigests = pickle.load(fp"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""
