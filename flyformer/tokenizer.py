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
        gene_tdigest_file: Path = None,
        gene_approx_cdf_file: Path = None,
        gene_approx_cdf_nsample: int = 1001,
        gene_percentile_cutoffs: list = GENE_PERCENTILE_CUTOFFS,
        embedding_size: int = EMBEDDING_SIZE,
        custom_attr_name_dict: dict = None,
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

        if gene_approx_cdf_file:
            logger.info(f"Loading approximated CDFs: {gene_approx_cdf_file}")
            self.gene_approx_cdfs = read_pickle(gene_approx_cdf_file)
            first_cdf = gene_approx_cdfs.values())[0]
            self.gene_approx_cdf_nsample = len(list(first_cdf)
        else:
            # Gene expression TDigest(s)
            logger.info(f"Loading TDigest(s): {gene_tdigest_file}")
            gene_tdigests = read_pickle(gene_tdigest_file)
            # Approximate CDF for gene token optimization
            logger.info(f"Approximating CDFs from TDigest(s). n = {self.gene_approx_cdf_nsample}")
            self.gene_approx_cdfs = self._approximate_gene_cdfs(gene_tdigests)

        logger.info("Generating token dictionary")
        self.gene_tokens = { "<pad>": 0, "<mask>": 1 }
        self._fill_gene_tokens()
        logger.info(f"Vocabulary size: {len(self.gene_tokens.keys())}")

        # Model input size (embedding)
        self.embedding_size = embedding_size

        # # gene keys for full vocabulary
        # gene_keys = list(self.gene_percentiles.keys())

        # # protein-coding and miRNA gene list dictionary for 
        # # selecting .loom rows for tokenization
        # self.genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))

    def _approximate_gene_cdfs(self,
            gene_tdigests: dict,
       ) -> None:
        """
        Compute gene approximate CDF based on tdigest file and
        `self.gene_approx_cdf_nsample`

        Parameters
        ----------
        gene_tdigests: dict
            Dictionary containing gene TDigests acrossflyformer-corpus.
            { genename: <TDigest> }
        """

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

    def _get_gene_percentile_cutoff(cdf: float) -> int:
        """
        Retrieve percentile cutoff associated to CDF

        Parameters
        ----------
        cdf: float
            CDF value corresponding to log1p gene expression of interest

        Returns
        ----------
        percentile_cutoff: int
            Percentile cut-off greater or equal to provided `cdf`
        """
        idx = np.argmax(self.gene_percentile_cutoffs >= cdf)
        return self.gene_percentile_cutoffs[idx]

    def _approximate_gene_cdf(self, gene: str, expression: float) -> float:
        """
        Approximate TDigest.cdf() to optimize tokenization
        `self.gene_cdfs` contains approximated CDFs based on with resolution
        `self.gene_approx_cdf_nsample`

        Parameters
        ----------
        gene: str
            Flybase genename. Key in `self.gene_tdigests` and
            `self.gene_approx_cdfs`
        expression: float
            log1p normalized gene expression value

        Returns
        ----------
        cdf: float
            Approximate CDF value (analogous to TDigest.cdf(). i.e., percentile
            associated to a provided expression value
        """
        gene_cdf = self.gene_cdfs.get(gene, np.array([np.inf]))
        cdf_idx = np.argmax(gene_cdf)
        cdf_percentile = (cdf_idx / self.gene_approx_cdf_nsample) * 100
        return cdf_percentile

    def tokenize_gene(
            self, 
            gene: str,
            expression: float
        ) -> Tuple[float, float]:
        """
        Return expression CDF value approximated from gene's expression data
        stored in TDigest and gene token (based on cdf and 
        `self.gene_percentile_cutoffs`)

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
        cdf = self._approximate_gene_cdf(gene, expression)
        percentile_cutoff = self._get_gene_percentile_cutoff(cdf)
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
                tokenized_cells.append(self.tokenize_cell(cell_data, genes))
                break

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
