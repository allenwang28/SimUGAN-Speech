# -*- coding: utf-8 *-* 
"""SyntheticSpeech Data Generator

TODO - Figure out what to do for test data
"""

from SimUGANSpeech.definitions import SYNTHETIC_DIR
from SimUGANSpeech.data import SpeechBatchGenerator, DEFAULT_BATCH_SIZE, DEFAULT_MAX_TIME_STEPS
from SimUGANSpeech.data import DEFAULT_MAX_OUTPUT_LENGTH, DEFAULT_BATCH_SIZE, ACCEPTED_LABELS
from SimUGANSpeech.data import FEATURES, DEFAULT_CHUNK_PROCESS_PERCENTAGE, DEFAULT_VALIDATION_PCT

class SyntheticSpeechBatchGenerator(SpeechBatchGenerator):
    def __init__(self,
                 features,
                 feature_sizes,
                 validation_pct=DEFAULT_VALIDATION_PCT,
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
            validation_pct (:obj:`float`, optional): The percent of training
                data to be held back for validation.
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
                         [], # TODO
                         features,
                         feature_sizes,
                         validation_pct=validation_pct,
                         audio_params=audio_params,
                         batch_size=batch_size,
                         chunk_pct=chunk_pct,
                         verbose=verbose)

