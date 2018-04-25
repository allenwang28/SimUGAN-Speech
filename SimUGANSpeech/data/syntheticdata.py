# -*- coding: utf-8 *-* 
"""SyntheticSpeech Data Generator

"""

from SimUGANSpeech.definitions import SYNTHETIC_DIR
from SimUGANSpeech.data import SpeechBatchGenerator
from SimUGANSpeech.data import DEFAULT_BATCH_SIZE
from SimUGANSpeech.data import DEFAULT_MAX_TIME_STEPS
from SimUGANSpeech.data import DEFAULT_MAX_OUTPUT_LENGTH
from SimUGANSpeech.data import DEFAULT_BATCH_SIZE
from SimUGANSpeech.data import ACCEPTED_LABELS
from SimUGANSpeech.data import FEATURES
from SimUGANSpeech.data import DEFAULT_CHUNK_PROCESS_PERCENTAGE 

class SyntheticSpeechBatchGenerator(SpeechBatchGenerator):
    def __init__(self,
                 features,
                 feature_sizes,
                 audio_params=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 chunk_pct=DEFAULT_CHUNK_PROCESS_PERCENTAGE,
                 verbose=True):
        """LibriSpeechBatchGenerator class initializer
        
        Args:
            features (list of str): List of desired features.
                See constant defined FEATURES for list of valid features.
            feature_sizes (list of int): List of maximum length of features.
                Has to be the same shape as features. The features will be
                truncated or padded to match the specified shape.
                If no maximum/truncation desired, just provide None
            audio_params (:obj:`AudioParameters`, optional): Parameters for audio
                See /preprocessing/audio.py for more information.
                Defaults to None
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
        super().__init__(SYNTHETIC_DIR,
                         ['.'],
                         features,
                         feature_sizes,
                         audio_params,
                         batch_size,
                         chunk_pct,
                         verbose)

