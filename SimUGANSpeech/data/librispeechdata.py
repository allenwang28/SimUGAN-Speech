# -*- coding: utf-8 *-* 
"""LibriSpeech Data Generator

This module is used to provide a batch generator for 
LibriSpeech data to train our models.

"""

from SimUGANSpeech.definitions import LIBRISPEECH_DIR
from SimUGANSpeech.data import SpeechBatchGenerator
from SimUGANSpeech.data import DEFAULT_BATCH_SIZE
from SimUGANSpeech.data import DEFAULT_MAX_TIME_STEPS
from SimUGANSpeech.data import DEFAULT_MAX_OUTPUT_LENGTH
from SimUGANSpeech.data import DEFAULT_BATCH_SIZE
from SimUGANSpeech.data import ACCEPTED_LABELS
from SimUGANSpeech.data import FEATURES
from SimUGANSpeech.data import DEFAULT_CHUNK_PROCESS_PERCENTAGE 



POSSIBLE_FOLDERS = [
                     'dev-clean',
                     'dev-other',
                     'test-clean',
                     'test-other',
                     'train-clean-100',
                     'train-clean-360',
                     'train-other-500',
                   ]

class LibriSpeechBatchGenerator(SpeechBatchGenerator):
    def __init__(self,
                 folder_names,
                 features,
                 feature_sizes,
                 batch_size=DEFAULT_BATCH_SIZE,
                 chunk_pct=DEFAULT_CHUNK_PROCESS_PERCENTAGE,
                 verbose=True):
        """LibriSpeechBatchGenerator class initializer
        
        Args:
            folder_paths (list of str): List of the folder names (or datasets)
                (e.g., dev-clean, dev-test, etc.)
            features (list of str): List of desired features.
                See constant defined FEATURES for list of valid features.
            feature_sizes (list of int): List of maximum length of features.
                Has to be the same shape as features. The features will be
                truncated or padded to match the specified shape.
                If no maximum/truncation desired, just provide None
            batch_size (:obj:`int`, optional): The desired batch size.
                Defaults to 10
            chunk_pct (:obj:`float`, optional): The percentage of chunks to
                load into memory at a time. 
                The lower the value, the less likely to reach a memory error.
                The higher the value, the more efficient the batch generator
                Defaults to 0.3
            verbose (:obj:`bool`, optional): Whether or not to print statements.
                Defaults to True.
        
        """
        super().__init__(LIBRISPEECH_DIR,
                         folder_names,
                         features,
                         feature_sizes,
                         batch_size,
                         chunk_pct,
                         verbose)

