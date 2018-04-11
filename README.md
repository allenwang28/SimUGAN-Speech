# SimUGAN-Speech
Using technology from https://arxiv.org/abs/1612.07828 to generate realistic speech 

## Requirements

- Python 3.5
- Tensorflow 1.4.0
- numpy
- pandas
- urllib
- deco
- ...

# Usage

First, from the root directory, run

    $ pip install -e .

To download LibriSpeech data, 

    $ cd SimUGANSpeech/scripts
    $ python librispeech_initialize.py



Some code used as references:
- https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
- https://github.com/carpedm20/simulated-unsupervised-tensorflow
