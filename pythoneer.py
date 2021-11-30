
import torch
import numpy as np


"""

Python is great when typing is used: https://docs.python.org/3/library/typing.html

"""

def add(a: int, b: int, verbose: bool = False) -> int:
    """
    Sum two numbers. 

    :param a: first number for sum
    :param b: second number for sum
    :param verbose: print sum if it is True

    :return: sum of the two numbers

    """
    result = a + b

    if verbose: 
        print(result)

    return result


def add_np_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Sum two numpy arrays.

    :param a: first  numpy array
        :shape: (n, m)
    :param b: second numpy array
        :shape: (n, m)

    :return: sum of the two numpy arrays
        :shape: (n, m)
    
    """

    return np.sum(a, b)


def add_torch_array(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    Sum two torch arrays.

    :param a: first  torch array
        :shape: (n, m)
    :param b: second torch array
        :shape: (n, m)

    :return: sum of the two torch arrays
        :shape: (n, m)
    
    """

    return torch.sum(a, b)


