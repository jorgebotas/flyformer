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
                               "organ_major": "organ_major"})
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

import argparse
from datasets import Dataset
import loompy as lp
import logging
import numpy as np
from pathlib import Path
import pickle
import ray
from tdigest import TDigest
from tqdm import tqdm
from typing import Tuple
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

logger = logging.getLogger(__name__)

GENE_PERCENTILE_FILE = Path(__file__).parent / "gene_percentile_dictionary.pkl"
GENE_PERCENTILE_CUTOFFS = [ 10, 25, 75, 90, 100 ]
EMBEDDING_SIZE = 2**11
# Create a TDigest that always returns a CDF of 0 (only value is infinity)
TDIGEST_INF = TDigest()
TDIGEST_INF.update("inf")


def read_pickle(path: Path) -> dict:
    """
    Read pickle file and returns dictionary. Pickle file might be chunked
    path: Path
        Path to pickle file
    Returns: dictionary containing content in pickle file
    """
    dictionary = {}
    with open(path, "rb") as fp:
        while True:
            try:
                dictionary = { **dictionary, **pickle.load(fp) }
            except EOFError:
                break
        return dictionary


class TranscriptomeTokenizer:
    def __init__(
        self,
        gene_tdigest_file: Path,
        custom_attr_name_dict: dict = None,
        gene_percentile_file: Path = GENE_PERCENTILE_FILE,
        gene_percentile_cutoffs: list = GENE_PERCENTILE_CUTOFFS,
        embedding_size: int = EMBEDDING_SIZE,
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
        gene_tdigest_file : Path
            Path to pickle file containing dictionary of TDigest(s) of
            gene expression based on whole flyformer-corpus. The gene
            percentile dictionary will be created based on this object.
            { gene: <TDigest> ... }
        embedding_size: int
            LLM model input size (embedding size). Default 2^11 = 2048
            Cell embeddings will be truncated if it exceeds this number
        """
        # dictionary of custom attributes 
        # {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # percentile cutoffs
        self.gene_percentile_cutoffs = gene_percentile_cutoffs

        # Gene expression TDigest(s)
        logger.info(f"Loading TDigest(s) from: {gene_tdigest_file}")
        self.gene_tdigests = read_pickle(gene_tdigest_file)

        # if gene_tdigest_file is not None:
            # # Compute dictionary of gene expression percentile cutoffs 
            # # based on gene TDigests of normalized expression across corpus
            # self._compute_gene_percentiles_from_tdigest(gene_tdigest_file)
        # else:
            # # load dictionary of gene expression percentiles (cutoffs)
            # self.gene_percentiles = read_pickle(gene_percentile_file)

        logger.info("Generating token dictionary")
        self.gene_tokens = { "<pad>": 0, "<mask>": 1 }
        self._fill_gene_tokens()

        # Model input size (embedding)
        self.embedding_size = embedding_size

        # # gene keys for full vocabulary
        # gene_keys = list(self.gene_percentiles.keys())

        # # protein-coding and miRNA gene list dictionary for 
        # # selecting .loom rows for tokenization
        # self.genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))

    def _compute_gene_percentiles_from_tdigest(self, 
            gene_tdigest_file: Path
        ) -> None:
        """
        Compute gene percentiles based on tdigest files

        Parameters
        ----------
        gene_tdigest_file: Path
            Path to pickle file containing gene TDigests across 
            flyformer-corpus.
            { genename: TDigest }
        """
        gene_tdigests = read_pickle(gene_tdigest_file)
        self.gene_percentiles = { 
            gene: [ 
                [ p, tdig.percentile(p) ] for p in self.gene_percentile_cutoffs 
            ] for gene, tdig in gene_tdigests.items()
        }

    def _fill_gene_tokens(self) -> None:
        """Fills self.gene_tokens with data from self.gene_percentiles"""
        idx = len(self.gene_tokens.keys())
        for gene in self.gene_tdigests.keys():
            for percentile in self.gene_percentile_cutoffs:
                self.gene_tokens[f"{gene}_{percentile}"] = idx
                idx += 1

    def tokenize_gene(
            self, 
            gene: str,
            expression: float
        ) -> Tuple[float, float]:
        """
        Return expression CDF value associated to gene's expression in TDigest
        and gene token (based on cdf and `self.gene_percentile_cutoffs`)

        Parameters
        ----------
        gene: str
            Gene name (Flybase). Keys in `self.gene_tdigests`
        expression: float
            1e4 normalized and log1p transformed gene expression
        Returns
        ----------
        cdf: float
            Cumulative Distribution Function value associated to gene
            expression when compared to whole corpus (TDigest)
        token: int
            Token associated to gene and expression level (based on 
            `self.gene_percentile_cutoffs`). See `self.gene_tokens`
        """
        tdigest = self.gene_tdigests.get(gene, TDIGEST_INF)
        cdf = tdigest.cdf(expression)
        percentile_cutoff = next(
            pc for pc in self.gene_percentile_cutoffs if pc / 100 >= cdf
        )
        token = self.gene_tokens.get(f"{gene}_{percentile_cutoff}", None)
        return cdf, percentile_cutoff, token


    def tokenize_cell(self, cell_data: np.array, genes: np.array) -> np.array:
        """
        Convert normalized gene expression vector to tokenized rank value 
        encoding.

        Parameters
        ----------
        """
        # Limit tokenization to non-zero expression genes
        nonzero_mask = cell_data > 0
        nonzero_genes = genes[nonzero_mask]
        nonzero_expression = cell_data[nonzero_mask]
        
        cdf_expression, pc, gene_token = zip(*[
            self.tokenize_gene(gene, expr)
                for gene, expr in zip(nonzero_genes, nonzero_expression)
        ])
        print(list(zip(genes, cdf_expression, pc))[0:20])
        return
        # create array of gene vector with token indices
        # mask undetected genes
        nonzero_mask = np.nonzero(gene_vector)[0]
        # sort by median-scaled gene values
        sorted_indices = np.argsort(-gene_vector[nonzero_mask])
        # tokenize
        sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
        return sentence_tokens

    # @ray.remote
    def tokenize_file(self, loom_file_path):
        # if self.custom_attr_name_dict is not None:
            # file_cell_metadata = {
                # attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            # }

        tqdm_desc = f"Tokenizing {loom_file_path}"
        with lp.connect(str(loom_file_path)) as data:
            genes = np.array(data.ra.var_names)
            cells = np.array(data.ca.obs_names)
            for idx, cell in enumerate(tqdm(cells, desc=tqdm_desc)):
                cell_data = data[:, idx]
                self.tokenize_cell(cell_data, genes)
                return

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


            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

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

        return tokenized_cells #, file_cell_metadata

    def tokenize_files(self, loom_data_directory: Path) -> list:
        """

        Parameters
        ----------
        """
        loom_files = list(Path(loom_data_directory).glob("*.loom"))
        
        if len(loom_files) == 0:
            logger.error(
                f"No .loom files found in directory {loom_data_directory}.")
            raise

        self.tokenize_file(loom_files[0])
        return

        # Initialize Ray
        ray.init()

        # Will store list of tokenized cells
        tokenized_cells = []

        # Loop through directories to tokenize .loom files in parallel (Ray)
        for file_tokenized_cells in ray.get([
                self.tokenize_file.remote(self, path) for path in loom_files
            ]):
            tokenized_cells += file_tokenized_cells

        # Shutdown Ray
        ray.shutdown()

        return tokenized_cells

    def create_dataset(self, tokenized_cells, cell_metadata, nproc=1):
        """

        Parameters
        ----------
        """
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        def truncate(example):
            example["input_ids"] = example["input_ids"][0:self.embedding_size]
            return example

        output_dataset_truncated = output_dataset.map(truncate, num_proc=nproc)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=nproc
        )

        return output_dataset_truncated_w_length


def parse_args():
    parser = argparse.ArgumentParser(
            "Tokenize cell data based on expression values")
    parser.add_argument("--tdigests", "-t", required=True, 
                        type=argparse.FileType('r'))
    parser.add_argument("--loom", "-l", required=True,
                        type=argparse.FileType('r'))
    parser.add_argument("--token_dict", "-t", required=True,
                        type=argparse.FileType('r'))
    parser.add_argument("--output", "-o", required=True,
                        type=argparse.FileType('w'))
    return parser.parse_args()
