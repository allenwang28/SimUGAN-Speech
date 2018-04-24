# -*- coding: utf-8 *-* 
"""Helper functions for data operations

This module is used to provide helper functions for any data related
operations. Included functions:

    - chunkify
    - randomly_sample_stack
    - pad_or_truncate

Todo:

"""
import numpy as np
import copy

from deco import concurrent, synchronized

def chunkify(l, num_chunks):
    """Break a list into N different lists of equal size

    Notes:
        Right now, this is just a wrapper for numpy's array_split
        function. We use this in case we ever want to change the 
        implementation

    Args:
        l (list): The list to be chunked
        num_chunks (int): The number of chunks to create

    Returns:
        list of lists: The chunkified list

    """
    return np.array_split(l, num_chunks)


def randomly_sample_stack(stack, N):
    """Randomly sample N elements without replacement

    Args:
        stack (list): The list of elements
        N (int): The number of elements to sample

    Returns:
        list: list of N elements

    Notes:
        If N > num elements in stack, then we return
        whatever is left in the stack

    """
    np.random.shuffle(stack)
    if len(stack) > N:
        return [stack.pop() for i in range(N)] 
    else:
        res = copy.deepcopy(stack)
        stack[:] = [] # Empty the stack by reference
        return res 

@synchronized
def pad_or_truncate(data, length):
    """Pads or truncates a list of data

    For padding,
    - numpy arrays are padded by 0s
    - strings are padded by spaces

    Args:
        data (list): The list of features

    Returns:
        list: The features padded or truncated
    """
    assert type(data) == list
    assert len(data) > 0

    @concurrent
    def pad_or_truncate_sample(sample, length):
        sample_type = type(sample)
        if sample_type == np.ndarray:
            new_shape = [length] + list(sample.shape[1:])
            res = np.zeros(new_shape)
            m = min(length, len(sample))
            res[:m] = sample[:m]
            return res
        elif sample_type == str:
            return sample[:length] + ' ' * (length - len(sample))
        else:
            raise ValueError("Cannot pad type {0}".format(sample_type))

    rmap = {} 
    for i, sample in enumerate(data):
        rmap[i] = pad_or_truncate_sample(sample, length)

    return [rmap[i] for i in range(len(rmap))]
