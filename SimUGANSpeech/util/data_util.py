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
import tensorflow as tf

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


def randomly_split_list_to_sizes(l, sizes_pct):
    """Randomly split a list into different sizes

    Args:
        l (list): The list to be split
        sizes_pct (list of floats): A list of all
            the percentage sizes. Should sum < 1

    Returns:
        A list of lists
    """
    l = copy.deepcopy(l)
    num_elements = len(l)
    result = []
    for s in sizes_pct:
        N = int(np.ceil(s * num_elements))
        result.append(randomly_sample_stack(l, N))
    result.append(l)
    return result


def randomly_split(l, first_pct_size):
    """Randomly split a list into two

    Args:
        l (list): The list to be split
        first_pct_size (float): Percentage of 
            size for first list returned

    Returns:
        tuple of lists: two lists, of sizes:
            (first_pct_size * N, (1 - first_pct_size) * N)
    """
    return randomly_split_list_to_sizes(l, [first_pct_size])


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


def letter_to_id(letter):
    if letter == ' ':
        return 27
    if letter == '\'':
        return 26
    return ord(letter) - ord('A')


def text_to_indices(text): 
    """Convert a string to a list of indices"""
    return [letter_to_id(letter) for letter in text.upper()]


def one_hot_transcriptions(transcriptions, vocabulary_size):
    """One hot encode transcriptions"""
    t_idx = np.array([text_to_indices(transcription) for transcription in transcriptions])
    return tf.one_hot(t_idx, vocabulary_size, dtype=tf.uint8)


