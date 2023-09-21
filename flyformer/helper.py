import pickle
from types import FunctionType

def inherit_doc(cls: class) -> class:
    """
    Decorator function to make all concrete methods of subclasses inherit
    docstring from parent's methods, if docstring is not specified.

    @inherit_doc
    class Class:
        def method
            ...
        ...
    Parameters
    ----------
    cls: class

    Returns
    ----------
    cls: class
        Class with "ammended" docstrings
    """
    for name, func in vars(cls).items():
        if isinstance(func, FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls

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
