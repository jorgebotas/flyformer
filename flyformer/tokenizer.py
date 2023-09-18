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
from tqdm import tqdm
from typing import Tuple
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDING_SIZE = 2**11


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
                dictionary.update(pickle.load(fp))
            except EOFError:
                break
        return dictionary


class TranscriptomeTokenizer:
    def __init__(
            self,
            gene_tdigest_file: Path = None,
            gene_approx_cdf_file: Path = None,
            gene_approx_cdf_nsample: int = 1001,
            gene_percentile_cutoffs: np.ndarray = np.array([ 10, 25, 75,
                                                             90, 100 ]),
            embedding_size: int = EMBEDDING_SIZE,
            custom_attr_name_dict: dict = None,
        ) -> None:
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
            Path to pickle file containing dictionary of tdigest.TDigest(s) of
            gene expression based on whole flyformer-corpus. The gene
            percentile dictionary will be created based on this object.
            { gene: <TDigest> ... }
        gene_approx_cdf_file : Path
            Path to pickle file containing dictionary with approximations of
            gene expression values in TDigest(s) computed on whole
            flyformer-corpus. Gene tokenization will be based on this object.
            { gene: <np.array> ... }
        embedding_size: int
            LLM model input size (embedding size). Default 2^11 = 2048
            Cell embeddings will be truncated if it exceeds this number
        """
        # percentile cutoffs
        self.gene_percentile_cutoffs = np.array(gene_percentile_cutoffs)

        if gene_approx_cdf_file is not None:
            print(f"Loading approximated CDFs: {gene_approx_cdf_file}")
            self.gene_approx_cdfs = read_pickle(gene_approx_cdf_file)
            first_cdf = list(self.gene_approx_cdfs.values())[0]
            # Check whether approximated CDFs are in correct format
            if not type(first_cdf) == np.ndarray:
                logger.warning(f"Approximated CDF must be of type np.ndarray!")
                print(f"Converting gene CDFs to np.ndarray...")
                self.gene_approx_cdfs = {
                    gene: np.array(arr) for gene, arr in self.gene_approx_cdfs
                }
            self.gene_approx_cdf_nsample = len(first_cdf)
        else:
            # Gene expression TDigest(s)
            print(f"Loading TDigest(s): {gene_tdigest_file}")
            gene_tdigests = read_pickle(gene_tdigest_file)
            # Approximate CDF for gene token optimization
            self.gene_approx_cdf_nsample = gene_approx_cdf_nsample
            print(f"Approximating CDFs from TDigest(s).\
                    n = {self.gene_approx_cdf_nsample}")
            self.gene_approx_cdfs = self._approximate_gene_cdfs(gene_tdigests)

        self.gene_tokens = { "<pad>": 0, "<mask>": 1 }
        self._fill_gene_tokens()
        print(f"Vocabulary size: {len(self.gene_tokens.keys())}")

        # Model input size (embedding)
        self.embedding_size = embedding_size

        # dictionary of custom attributes
        # {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # # gene keys for full vocabulary
        # gene_keys = list(self.gene_percentiles.keys())

        # # protein-coding and miRNA gene list dictionary for
        # # selecting .loom rows for tokenization
        # self.genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))

    def _approximate_gene_cdfs(self,
            gene_tdigests: dict,
       ) -> None:
        """
        Compute gene approximate CDF based on tdigest file. Sample each gene
        expression distribution `self.gene_approx_cdf_nsample` times, equally
        interspered from 0 to 100th percentile (inclusive)

        Parameters
        ----------
        gene_tdigests: dict
            Dictionary containing gene TDigests acrossflyformer-corpus.
            { genename: <TDigest> }

        Returns
        ----------
        gene_approximated_cdfs: dict
            Dictionary containing an np.ndarray for each gene with an
            approximation of the gene's expression distribution in the TDigest.
            { genename: <np.ndarray> }
            where len(np.ndarray) = self.gene_approx_cdf_nsample (odd number)
        """
        # Make sure `self.gene_approx_cdf_nsample` is an odd number to include
        # min and maximum log1p gene expression values
        if self.gene_approx_cdf_nsample % 2 == 0:
            self.gene_approx_cdf_nsample += 1

        # Converts to percentile (input of `TDigest.percentile()`)
        factor = 100 / (self.gene_approx_cdf_nsample - 1)

        tqdm_desc = f"CDF approximation (n={self.gene_approx_cdf_nsample})"
        gene_approximated_cdfs = {
            gene: np.array([
                tdigest.percentile(i * factor)
                for i in range(self.gene_approx_cdf_nsample)
            ]) for gene, tdigest in tqdm(gene_tdigests.items(), desc=tqdm_desc)
        }
        print(len(list(gene_approx_cdfs.values())[0]))
        return gene_approximated_cdfs

    def _fill_gene_tokens(self) -> None:
        """Fills self.gene_tokens with data from self.gene_percentiles"""
        idx = len(self.gene_tokens.keys())
        for gene in self.gene_approx_cdfs.keys():
            for percentile in self.gene_percentile_cutoffs:
                self.gene_tokens[f"{gene}_{percentile}"] = idx
                idx += 1

    def _get_gene_percentile_cutoff(self, cdf: float) -> int:
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
        idx = np.argmax(self.gene_percentile_cutoffs >= cdf * 100)
        return self.gene_percentile_cutoffs[idx]

    def _approximate_gene_cdf(self, gene: str, expression: float) -> float:
        """
        Approximate TDigest.cdf() to optimize tokenization
        `self.gene_approx_cdfs` contains approximated CDFs based on with
        resolution `self.gene_approx_cdf_nsample`

        Parameters
        ----------
        gene: str
            Flybase genename. Keys in `self.gene_approx_cdfs`
        expression: float
            log1p normalized gene expression value

        Returns
        ----------
        cdf: float [0, 1]
            Approximate CDF value (analogous to TDigest.cdf(). i.e., percentile
            associated to a provided expression value
        """
        # gene_cdf is ordered from low to high
        gene_cdf = self.gene_approx_cdfs.get(gene, np.array([np.inf]))

        # Find index of first value greater or equal to expreesion value
        cdf_idx = np.argmax(gene_cdf >= expression)

        # Equivalent to cdf_idx / len(gene_cdf)
        cdf = (cdf_idx / self.gene_approx_cdf_nsample)

        return cdf

    def _tokenize_gene(
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
            Gene name (Flybase). Keys in `self.gene_approx_cdfs`
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
        # Obtain approximate CDF (0-100)
        cdf = self._approximate_gene_cdf(gene, expression)

        percentile_cutoff = self._get_gene_percentile_cutoff(cdf)

        # Obtain token
        token = self.gene_tokens.get(f"{gene}_{percentile_cutoff}", -1)

        return cdf, token

    def _tokenize_cell(
            self,
            cell_data: np.ndarray,
            genes: np.ndarray
        ) -> np.ndarray:
        """
        Converts normalized gene expression vector to tokenized rank value
        encoding.Relative gene expression value is calculated based
        on gene's approximated cdf obtained from TDigest.
            e.g. 0.93 cdf > 0.81 cdf > 0.13 cdf

        Parameters
        ----------
        cell_data: np.ndarray
            Log1p-transformed gene expression data for single cell
        genes: np.ndarray
            Genes to be tokenized based on cell expression data

        Returns
        ----------
        gene_tokens: np.ndarray
            Sorted gene tokens in ascending order of relative gene expression
            across corpus. Output is truncated to a maximum length equal to
            `self.embedding_size` (maximum model input size)
        """
        # Limit tokenization to non-zero expression genes
        nonzero_mask = cell_data > 0
        nonzero_genes = genes[nonzero_mask]
        nonzero_expression = cell_data[nonzero_mask]

        # Tokenize each gene with non-zero expression in cell data
        cdf_expression, gene_tokens = zip(*[
            self._tokenize_gene(gene, expr)
                for gene, expr in zip(nonzero_genes, nonzero_expression)
        ])

        # Convert to np.ndarray
        cdf_expression = np.array(cdf_expression)
        gene_tokens = np.array(gene_tokens)

        # Obtain indices from descending sorted cdf expression values
        sorted_indices = np.argsort(-cdf_expression)

        # Sort tokens by descending cdf expression
        sorted_gene_tokens = gene_tokens[sorted_indices]
        # Return truncated gene_tokens to match max model input size
        return sorted_gene_tokens # sorted_gene_tokens[:self.embedding_size]

    @ray.remote
    def _tokenize_file(self, loom_file_path: Path) -> list:
        """
        Tokenize single cell gene expression data from preprocessed loom file
        See `flyformer.data_preprocessing` to properly format .loom files PRIOR
        to tokenization. File tokenization is parallelized using Ray.

        Call `self._tokenize_file.remote(...)`

        Parameters
        ----------
        loom_file_path: Path
            Path to preprocessed loom file

        Returns
        ----------
        tokenized_cells: list
            Matrix containing tokenized cells in `loom_file_path`
        """
        # if self.custom_attr_name_dict is not None:
            # file_cell_metadata = {
                # attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            # }

        tokenized_cells = []
        tqdm_desc = f"Tokenizing {loom_file_path}"
        with lp.connect(str(loom_file_path)) as data:
            genes = np.array(data.ra.var_names)
            cells = np.array(data.ca.obs_names)

            # Tokenize each cell in .loom file
            for idx, cell in enumerate(tqdm(cells, desc=tqdm_desc)):
                cell_data = data[:, idx]
                tokenized_cells.append(self._tokenize_cell(cell_data, genes))

                # # add custom attributes for subview to dict
                # if self.custom_attr_name_dict is not None:
                    # for k in file_cell_metadata.keys():
                        # file_cell_metadata[k] += subview.ca[k].tolist()
                # else:
                    # file_cell_metadata = None

        return tokenized_cells #, file_cell_metadata

    def _tokenize_files(self, loom_data_directory: Path) -> list:
        """
        Tokenize single cell gene expression data from a collection of
        preprocessed loom files contained in `loom_data_directory`. See
        `flyformer.data_preprocessing` to properly format .loom files PRIOR to
        tokenization. File tokenization is parallelized using Ray.

        Parameters
        ----------
        loom_data_directory: Path
            Path to directory containing preprocessed loom files

        Returns
        ----------
        tokenized_cells: list
            Matrix containing tokenized cells
        """
        loom_files = list(Path(loom_data_directory).glob("*.loom"))

        if len(loom_files) == 0:
            logger.error(
                f"No .loom files found in directory {loom_data_directory}.")
            raise

        # Initialize Ray
        ray.init()

        # Will store list of tokenized cells
        tokenized_cells = []

        # Loop through directories to tokenize .loom files in parallel (Ray)
        for file_tokenized_cells in ray.get([
                self._tokenize_file.remote(self, path) for path in loom_files
            ]):
            tokenized_cells += file_tokenized_cells

        # Shutdown Ray
        ray.shutdown()

        return tokenized_cells

    def _create_dataset(self,
            tokenized_cells: list,
            cell_metadata: list,
            nproc: int = 1
        ) -> Dataset:
        """
        Create datasets.Dataset from tokenized cells

        Parameters
        ----------
        tokenized_cells: list
            Matrix of tokenized cells. Each cell is represented as a set of
            `self.gene_tokens` sorted in decreasing order or relative
            expression across data corpus.
        cell_metadata: list
            Metadata associated to each cell. E.g. cell-type, organ,...
        nproc: int
            Number of processors used for .dataset formatting (truncating and
            calculating example length)

        Returns
        ----------
        dataset: datasets.Dataset
            Dataset containing tokenized cells
        """

        def truncate(example):
            example["input_ids"] = example["input_ids"][0:self.embedding_size]
            return example

        def measure_length(example):
            """Measure length of example (max = `self.embedding_size`"""
            example["length"] = len(example["input_ids"])
            return example

        # create dict to build dataset
        dataset_dict = {"input_ids": tokenized_cells}
        # if self.custom_attr_name_dict is not None:
            # dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        output_dataset_truncated = output_dataset.map(truncate, num_proc=nproc)

        output_dataset_w_length = output_dataset_truncated.map(
            measure_length, num_proc=nproc
        )

        return output_dataset_w_length

    def tokenize_data(self,
            loom_data_directory: Path,
            output_directory: Path,
            output_prefix: str,
            nproc: int = 1,
        ) -> None:
        """
        Tokenize single cell gene expression data from a collection of
        preprocessed .loom files contained in `loom_data_directory`. See
        `flyformer.data_preprocessing` to properly format .loom files PRIOR to
        tokenization. File tokenization is parallelized using Ray.

        Tokenized single cell gene expression data will be saved as .dataset in
        `output_directory` (Apache Arrow format)

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        nproc: int
            Number of processors used for .dataset formatting (truncating and
            calculating example length)
        """
        loom_data_directory = Path(loom_data_directory)
        output_directory = Path(output_directory)
        output_path = output_directory / output_prefix

        # Tokenize cells from files in `loom_data_directory`
        tokenized_cells = self._tokenize_files(loom_data_directory)
        cell_metadata = []

        # Create dataset
        tokenized_dataset = self._create_dataset(tokenized_cells,
                                                 cell_metadata,
                                                 nproc=nproc)

        # Save dataset to disk
        tokenized_dataset.save_to_disk(output_path.with_suffix(".dataset"))


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
