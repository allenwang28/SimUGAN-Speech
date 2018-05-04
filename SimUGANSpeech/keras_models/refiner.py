from keras.layers import Input, Dropout, Concatenate, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Model

def unet1d(input_shape, output_shape, num_filters):
    # Construct a U-Net Generator
    def conv1d(layer_input, filters, f_size=4, bn=True):
        """Layers for downsampling"""
        d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv1d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers for upsampling"""
        u = UpSampling1D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    d0 = Input(shape=input_shape)

    # Downsampling
    d1 = conv1d(d0, num_filters)
    d2 = conv1d(d1, num_filters*2)
    d3 = conv1d(d2, num_filters*4)
    d4 = conv1d(d3, num_filters*8)
    d5 = conv1d(d4, num_filters*8)
    d6 = conv1d(d5, num_filters*8)
    d7 = conv1d(d6, num_filters*8)

    # Upsampling
    u1 = deconv1d(d7, d6, num_filters*8)
    u2 = deconv1d(u1, d5, num_filters*8)
    u3 = deconv1d(u2, d4, num_filters*8)
    u4 = deconv1d(u3, d3, num_filters*4)
    u5 = deconv1d(u4, d2, num_filters*2)
    u6 = deconv1d(u5, d1, num_filters)

    u7 = UpSampling1d(size=2)(u6)
    output = Conv1D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output)

