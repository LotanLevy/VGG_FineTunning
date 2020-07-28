from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16




import tensorflow as tf


class PerceptualModel(NNInterface):
    def __init__(self, classes_num, input_size):
        super().__init__(classes_num, input_size)
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[0], 3))
        vgg_conv.summary()

        for layer in vgg_conv.layers[:]:
            layer.trainable = False

        self.__model = tf.keras.Sequential()
        self.__model.add(vgg_conv)
        self.__model.add(tf.keras.layers.Flatten())
        self.__model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.__model.add(tf.keras.layers.Dropout(0.5))
        self.__model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.__model.add(tf.keras.layers.Dropout(0.5))
        self.__model.add(tf.keras.layers.Dense(classes_num, activation='softmax'))

        self.__model.summary()



    def call(self, x, training=True):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)

    def freeze_status(self):

        # for i, layer in enumerate(self.__model.layers):
        #     if freeze_idx > i:
        #         layer.trainable = False

        for i, layer in enumerate(self.__model.layers):
            if i == 0:
                for layer in self.__model[0].layers[:]:
                    print("layer {} is trainable {}".format(layer.name, layer.trainable))
            else:
                print("layer {} is trainable {}".format(layer.name, layer.trainable))