"""
    Author: Chris (https://github.com/machrisaa), modified by Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: tensorflow implemention of VGG 16 and VGG 19 based on tensorflow-vgg16
"""

import os
import tensorflow as tf
import numpy as np
import inspect

data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_name = dir_path + "/../lib/descriptor/vgg16.npy"
weights_url = "https://www.dropbox.com/s/gjtfdngpziph36c/vgg16.npy?dl=1"


class VGG19(object):
    def __init__(self, vgg16_npy_path=None):
        global data

        if vgg16_npy_path is None:
            path = inspect.getfile(VGG19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, weights_name)

            if os.path.exists(path):
                vgg16_npy_path = path
            else:
                print("VGG16 weights were not found in the project directory")
                print("Please download the numpy weights file and place it in present directory")
                print("Download link: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM")
                print("Exiting the program..")
                exit(1)

        if data is None:
            data = np.load(vgg16_npy_path, encoding='latin1')
            self.data_dict = data.item()
            print("VGG net weights loaded")

        else:
            self.data_dict = data.item()

    def build(self, image):

        self.conv1_1 = self.__conv_layer(image, "conv1_1")
        self.conv1_2 = self.__conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.__avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.__conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.__conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.__avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.__conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.__conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.__conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.__avg_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.__conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.__conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.__conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.__avg_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.__conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.__conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.__conv_layer(self.conv5_2, "conv5_3")

        # self.data_dict = None

        return {'relu2_2':self._gram_matrix(self.conv2_2),
                'relu3_1':self._gram_matrix(self.conv3_1),
                'relu3_2':self._gram_matrix(self.conv3_2),
                'relu4_2':self._gram_matrix(self.conv4_2),
                'relu4_3':self._gram_matrix(self.conv4_3)}

    def __avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.__get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.__get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def __get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def __get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def __get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def _gram_matrix(self, tensor):

        shape = tensor.get_shape()

        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram

if __name__ == "__main__":
    net = VGG19('net/vgg19.npy')
    image=tf.placeholder(tf.float32,[None, 200,200, 3])
    collection = net.build(image)
