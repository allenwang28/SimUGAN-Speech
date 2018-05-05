import datetime
import os
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam

from SimUGANSpeech.keras_models import ModelFactory
from SimUGANSpeech.definitions import KERAS_DIR


MODEL_DIR = os.path.join(KERAS_DIR, 'SimUGANSpeech')

class SimUGANSpeech():
    def __init__(self, spectrogram_shape, learning_rate=0.0002, beta_1=0.5):
        try:
            self.load_models()
            print ("Loading models from checkpoint.")
        except:
            print ("Models not found.")


        # spectrogram shape: (time, num_frequencies)
        self.spectrogram_shape = spectrogram_shape
        self.T = spectrogram_shape[0]
        self.F = spectrogram_shape[1]


        # Calculate output shape of D (PatchGAN)
        num_patches = 2**4
        patch = self.T // num_patches
        self.disc_patch = (patch, 1)

        optimizer = Adam(learning_rate, beta_1)


        # Build and compile the discriminator
        mf = ModelFactory('spectrogram')
        num_refiner_filters = 64
        num_discriminator_filters = 64

        self.discriminator = mf.discriminator(spectrogram_shape, num_discriminator_filters)

        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------
        # Build the refiner
        self.refiner = mf.refiner(spectrogram_shape, spectrogram_shape, num_refiner_filters)

        # Input spectrograms and their conditioning spectrograms 
        input_A = Input(shape=spectrogram_shape)
        input_B = Input(shape=spectrogram_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.refiner(input_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, input_B])

        self.combined = Model(inputs=[input_A, input_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def get_model_paths(self, backup_num=None):
        if backup_num: 
            h5_extension = "{0}.h5".format(backup_num)
            json_extension = "{0}.json".format(backup_num)
        else:
            h5_extension = ".h5"
            json_extension = ".json"

        model_paths = []
        model_paths.append(os.path.join(KERAS_DIR, 'refiner_weights{0}'.format(h5_extension)))
        model_paths.append(os.path.join(KERAS_DIR, 'refiner_arch{0}'.format(json_extension)))
        model_paths.append(os.path.join(KERAS_DIR, 'discrim_weights{0}'.format(h5_extension)))
        model_paths.append(os.path.join(KERAS_DIR, 'discrim_arch{0}'.format(json_extension)))
        model_paths.append(os.path.join(KERAS_DIR, 'combined_weights{0}'.format(h5_extension)))
        model_paths.append(os.path.join(KERAS_DIR, 'combined_arch{0}'.format(json_extension)))
        return tuple(model_paths)


    def load_models(self, backup_num=None):
        rw, ra, dw, da, cw, ca = self.get_model_paths(backup_num)
        with open(ra, 'r') as f:
            self.refiner = model_from_json(f.read())
        self.refiner.load_weights(rw)

        with open(da, 'r') as f:
            self.discriminator = model_from_json(f.read())
        self.discriminator.load_weights(dw)

        with open(ca, 'r') as f:
            self.combined = model_from_json(f.read())
        self.combined.load_weights(cw)



    def save_models(self, backup_num=None):
        rw, ra, dw, da, cw, ca = self.get_model_paths(backup_num)

        self.refiner.save_weights(rw)
        with open(ra, 'w') as f:
            f.write(self.refiner.to_json())
        
        self.discriminator.save_weights(dw)
        with open(da, 'w') as f:
            f.write(self.discriminator.to_json())

        self.combined.save_weights(cw)
        with open(ca, 'w') as f:
            f.write(self.combined.to_json())


    def train(self, epochs, real_data_source, fake_data_source, batch_size=10, save_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i in range(fake_data_source.num_batches):
                real_batch = real_data_source.get_training_batch()
                fake_batch = fake_data_source.get_training_batch()

                real_batch = np.array(real_batch[0])
                fake_batch = np.array(fake_batch[0])

                # ---------------------
                #  Train Discriminator
                # ---------------------

                refined_batch = self.refiner.predict_on_batch(fake_batch)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([real_batch, fake_batch], valid)
                d_loss_fake = self.discriminator.train_on_batch([refined_batch, fake_batch], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([real_batch, fake_batch], [valid, real_batch])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
            if epoch % save_interval == 0:
                print ("Backing up models...")
                self.save_models(epoch)
        self.save_models()            


if __name__ == '__main__':
    from SimUGANSpeech.data import LibriSpeechBatchGenerator
    from SimUGANSpeech.data import SyntheticSpeechBatchGenerator

    training_folder_names = [ 'dev-clean' ]
    testing_folder_names = []

    features = [ 'spectrogram' ] 
    feature_sizes = [ 512 ]
    batch_size = 10
    verbose = True 
    chunk_pct = None
    num_epochs = 100
    validation_pct = 0.8

    spectrogram_shape = (feature_sizes[0], 256)
    librispeech = LibriSpeechBatchGenerator(training_folder_names,
                                            testing_folder_names,
                                            features,
                                            feature_sizes=feature_sizes,
                                            batch_size=batch_size,
                                            chunk_pct=chunk_pct,
                                            validation_pct=validation_pct,
                                            verbose=verbose)     

    syntheticspeech = SyntheticSpeechBatchGenerator(features,
                                                    feature_sizes=feature_sizes,
                                                    batch_size=batch_size,
                                                    chunk_pct=chunk_pct,
                                                    validation_pct=validation_pct,
                                                    verbose=verbose)  

    gan = SimUGANSpeech(spectrogram_shape)
    gan.train(epochs=3000,
              real_data_source=librispeech,
              fake_data_source=syntheticspeech,
              batch_size=batch_size,
              save_interval=50)
