# -*- coding: utf-8 *-* 
"""Helper functions for data operations

This module is used to provide helper functions for any data related
operations. Included functions:

    - chunkify

Todo:

"""
import numpy as np


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


if __name__ == "__main__":
    list1 = list(range(10))
    list2 = reversed(list(range(10)))

    print (chunkify(list1, 2))






