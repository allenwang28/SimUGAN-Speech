

class Refiner(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size


    def construct(self):
        def conv1d(layer_input, filters, f_size=4, bn=True):
            """Layers for downsampling"""
            d = Conv1D()

        def deconv1d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers for upsampling"""
