# -*- coding: utf-8 *-* 
"""LibriSpeech Data Generator

This module is used to provide a batch generator for 
LibriSpeech data to train our models.

"""

from SimUGANSpeech.data import SpeechBatchGenerator

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
    pass


