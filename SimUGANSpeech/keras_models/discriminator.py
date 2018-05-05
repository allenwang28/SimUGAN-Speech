from keras.layers import Input, BatchNormalization, Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def convdiscriminator1d(input_shape, num_filters):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    input_A = Input(shape=input_shape)
    input_B = Input(shape=input_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_inputs = Concatenate(axis=-1)([input_A, input_B])

    d1 = d_layer(combined_inputs, num_filters, bn=False)
    d2 = d_layer(d1, num_filters*2)
    d3 = d_layer(d2, num_filters*4)
    d4 = d_layer(d3, num_filters*8)

    validity = Conv1D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([input_A, input_B], validity)

