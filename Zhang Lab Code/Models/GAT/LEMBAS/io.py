"""
Read and write python objects.
"""

import pathlib
import pickle
from typing import Any


def read_pickled_object(file_name: str):
    """Read an object as a pickled file.

    Parameters
    ----------
    file_name : str
        'full/path/to/file.pickle'
    
    Returns
    -------
    pickled_object
        the pickled object
    """
    with open(file_name, 'rb') as handle:
        pickled_object = pickle.load(handle)
    return pickled_object

def write_pickled_object(object: Any, file_name: str) -> None:
    """Save an object as a pickled file.

    Parameters
    ----------
    object : Any
        object to save
    file_name : str
        'full/path/to/file.pickle'
    """
    if '.' in file_name:
        p = pathlib.Path(file_name)
        extensions = "".join(p.suffixes)
        file_name = str(p).replace(extensions, '.pickle')
    else:
        file_name = file_name + '.pickle'

    with open(file_name, 'wb') as handle:
        pickle.dump(object, handle)