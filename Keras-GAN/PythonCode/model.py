from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D,Lambda, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers import ZeroPadding2D, Deconvolution2D
from keras.callbacks import TensorBoard
import tensorflow as tf

class Models:
    """Contains all SGAN Models
    """
    def Generator(self, LatentVector, width, filters):
        #Use GPU
        with tf.device('/gpu:0'):
            #Size of latent matrix
            latent_vector_size = LatentVector

            latent_vector_input = Input(shape=[latent_vector_size])
            G = Dense(width*width*filters, kernel_initializer='glorot_normal')(latent_vector_input)
            G = Dropout(0.5)(G)

            G = BatchNormalization(axis=1)(G)
            G = LeakyReLU(0.2)(G)

            G = Reshape([width, width, filters])(G)
            G = UpSampling2D(size=(2, 2))(G)
            G = Conv2D(filters = filters//2, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(G)
            G = Dropout(0.5)(G)

            G = BatchNormalization(axis=1)(G)
            G = LeakyReLU(.2)(G)

            G = Conv2D(filters=width//4, kernel_size=(3, 3), padding='same')(G)
            G = Dropout(0.5)(G)
            G = BatchNormalization(axis=1)(G)
            G = LeakyReLU(.2)(G)
            G = Conv2D(filters = 1, kernel_size=(1, 1), padding='same')(G)
            generated_image = Activation('tanh')(G)

            model = Model(inputs=latent_vector_input, outputs=generated_image)
            model.compile(loss='binary_crossentropy', optimizer="ADAM")
            return model

    def Discriminator(self, ImageShape):
        #Use GPU
        with tf.device('/gpu:0'):
            InputImage = Input(ImageShape)
            D = Conv2D(filters=16, kernel_size=(5, 5), padding= 'same')(InputImage)
            D = LeakyReLU(.2)(D)
            D = AveragePooling2D(pool_size=(2, 2))(D)
            D = Conv2D(filters=32,  kernel_size=(5, 5), padding = 'same')(D)
            D = AveragePooling2D(pool_size=(2, 2))(D)
            D = Flatten()(D)
            D = Dense(256)(D)
            D = LeakyReLU(.2)(D)
            D = Dense(1)(D)
            D = Activation('sigmoid')(D)
            model = Model(inputs=InputImage, outputs=D)
            return model

    def GeneratorWithDiscriminator(self, Generator, Discriminator):
        #Freeze weights in Descriminator as part of minimax game
        #Use GPU
        with tf.device('/gpu:0'):
            Discriminator.trainable = False
            #Create Generator Trainer
            model = Sequential()
            model.add(Generator)
            model.add(Discriminator)
            return model
