from pathlib import Path
import pickle
from typing import Union


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

def write_pickle(obj: Union[list, dict], path: Path) -> None:
    """
    Write object to pickle at path

    Parameters
    ----------
    obj: Union[list, dict]
        Object to write
    path: Path
        Path to write pickle file
    """
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)
