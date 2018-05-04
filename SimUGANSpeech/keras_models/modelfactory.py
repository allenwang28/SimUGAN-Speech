from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from SimUGANSpeech.models import unet1d, convdiscriminator1d

data_types = ['spectrogram', 'image']

class ModelFactory(object):

    def __init__(self, data_type):
        self.data_type = data_type

    def discriminator(self, input_shape, num_filters):
        if self.data_type == 'spectrogram':
            return (input_shape, num_filters)

    def refiner(self, input_shape, output_shape, num_filters):
        if self.data_type == 'spectrogram':
            return unet1d(input_shape, output_shape, num_filters)
