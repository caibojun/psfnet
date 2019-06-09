import tensorflow as tf
import numpy as np
import collections
import vgg19


class FeatureLoss:
    def __init__(self):

        self.net = vgg19.VGG19('net/vgg19.npy')


    def _build_graph(self, init_image, feature_image):

        """ prepare data """
        # this is what must be trained
        self.a0 = feature_image
        self.x0 = init_image
        _, h, w, c = self.x0.get_shape().as_list()

        # get feature-layer-feature for feature loss
        self.As = self.net.build(self.a0)

        # get layer-values for x
        self.Fs = self.net.build(self.x0)

        """ compute loss """
        L_feature = 0
        #for id in self.Fs.keys():

        F = self.Fs["conv4_2"]
        # print(F.get_shape().as_list())
        # _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
        N = h * w  # product of width and height
        M = c  # number of filters

        # w = 1.0/len(self.Fs.keys())  # weight for this layer

        A = self.As[id]  # style feature of a

        F = tf.reshape(F, shape=[1, -1])
        A = tf.reshape(A, shape=[1, -1])
        # L_feature += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((F-A), 2))
        L_feature += (1. - tf.matmul(F, A, transpose_b=True)[0, 0] / (tf.norm(F) * tf.norm(A)))
        self.L_feature = L_feature
        return self.L_feature