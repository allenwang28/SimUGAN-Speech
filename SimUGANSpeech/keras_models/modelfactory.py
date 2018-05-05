from SimUGANSpeech.keras_models import unet1d, convdiscriminator1d

data_types = ['spectrogram', 'image']

class ModelFactory(object):

    def __init__(self, data_type):
        self.data_type = data_type

    def discriminator(self, input_shape, num_filters):
        if self.data_type == 'spectrogram':
            return convdiscriminator1d(input_shape, num_filters)

    def refiner(self, input_shape, output_shape, num_filters):
        if self.data_type == 'spectrogram':
            return unet1d(input_shape, output_shape, num_filters)
