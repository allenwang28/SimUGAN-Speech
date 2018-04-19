# -*- coding: utf-8 *-* 
"""Helper functions for data operations

This module is used to provide helper functions for any data related
operations. Included functions:

    - chunkify

Todo:

"""
import numpy as np
import copy

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