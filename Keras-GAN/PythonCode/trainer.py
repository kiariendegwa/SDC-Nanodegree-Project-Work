from sklearn.utils import shuffle
from PIL import Image
import scipy.stats as stats
import tensorflow as tf
from model import Models
from keras.datasets import mnist
import numpy as np
import math
import tensorflow as tf
import sys

class Trainer:
    """This class contains all the training packages needed for the GAN
    """
    def __init__(self):
        self.datasets = mnist.load_data()

    def Train(self, BatchSize, Epochs, LatentVectorSize):
        #Use GPU
        with tf.device('/gpu:0'):
            models = Models()
            #Set up discriminator
            discriminator = models.Discriminator(ImageShape=(28, 28, 1))
            discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
            discriminator.trainable = True

            #Set up generator
            generator = models.Generator(LatentVector=LatentVectorSize, width =14, filters=100)

            #Set up GAN
            GAN = models.GeneratorWithDiscriminator(Generator=generator, Discriminator=discriminator)
            GAN.compile(loss='binary_crossentropy', optimizer="ADAM")

            #Get Data
            (X_train, Y_train), (X_test, Y_test) = self.datasets

             #Get training data and shuffle
            X_train, Y_train = shuffle(X_train[:10000], Y_train[:10000])

            for Epoch in range(Epochs):
            #Training sequence: -> Generator -> Descriminator -> GAN
                print('Epoch: {}'.format(Epoch))

                for offset in range(0,X_train.shape[0], BatchSize):
                    end = offset + BatchSize
                    batch_X_train, batch_Y_train = X_train[offset:end], Y_train[offset:end]

                    #Use spherical noise between (-1, 1), i.e. Gaussian distribution not uniform noise
                    mu, sigma = 0, 1
                    lower, upper = -1, 1
                    noise_batch = np.asarray([stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,\
                                               loc=mu,scale=sigma,size=LatentVectorSize)\
                                        for i in range(BatchSize)])

                    #Generate images from Generator
                    GeneratedImages =  np.asarray(generator.predict(noise_batch, verbose=1))

                    #Save every image every 50 batches and 10 epochs
                    if offset%1000==0 and Epoch%10==0:
                        SpriteImg = self.SpriteGen(GeneratedImages)
                        Image.fromarray(SpriteImg, mode='RGB').save("TrainingImages/Epoch_"+str(Epoch)+"_"+\
                                                       "Batch_"+str(offset)+".png")
                    #Generate GAN training data
                    batch_X_train = batch_X_train.reshape(GeneratedImages.shape)
                    combined_X_train =  np.vstack((batch_X_train, GeneratedImages))

                    #If real data set values 0.7 and 1.2, and 0.0 for fake data: i.e. use soft labels
                    combined_Y_train = np.hstack((np.random.uniform(0.7, 1.2, BatchSize), \
                                                  np.asarray([0.0]*BatchSize)))

                    #Discriminator
                    d_loss = discriminator.train_on_batch(combined_X_train, combined_Y_train)

                    #Train on GAN
                    discriminator.trainable = False
                    g_loss = GAN.train_on_batch(noise_batch, np.ones(BatchSize))
                    discriminator.trainable = True

                    #Save weights every 5 batches
                    if  Epoch%5== 0:
                        generator.save('Generator.h5')
                        discriminator.save('Discriminator.h5')

                print("Discriminator Loss: {}".format(d_loss))
                print("Generator Loss: {}".format(g_loss))

    def SpriteGen(self, image_batches):
        #Convert to 8 bit matrix between 0 & 255
        image_batches = (image_batches*127.5+127.5).astype(np.uint8)

        number_of_images = image_batches.shape[0]

        #Height and width of final sprite image:
        height = int(math.sqrt(number_of_images))
        width = int(math.ceil(number_of_images/height))

        image_dim = image_batches.shape[1:]

        #Final output sprite image:s
        sprite = np.zeros((height*image_dim[0], width*image_dim[1]))

        for index, img in enumerate(image_batches):
            i = int(index/width)
            j = index % width
            sprite[i*image_dim[0]:(i+1)* image_dim[0], j*image_dim[1]:(j+1)*image_dim[1]] = \
                image_batches[0, :, :].reshape(image_dim[0], image_dim[1])
        return sprite

    def main(self):
        self.Train(BatchSize=200, Epochs=100, LatentVectorSize=100)

if __name__ == "__main__":
    Trainer = Trainer()
    Trainer.main()
