from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mobilenet(inputs,
          is_training=True,
          width_multiplier=1,
          reuse = False,
          scope='MobileNet'):
    """ MobileNet
    More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
    Returns:
    prediction: deblur image
    """
    def instance_norm(input, name="instance_norm"):
        with tf.variable_scope(name):
            depth = input.get_shape()[3]
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
        return scale*normalized + offset 

    def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

        bn = instance_norm(depthwise_conv, name=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
        bn = instance_norm(pointwise_conv, name=sc+'/pw_batch_norm')
        return bn
  
    def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, sc="deconv2d"):
        with tf.variable_scope(sc):
            deconv = tf.contrib.slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                            biases_initializer=None)
            bn = slim.batch_norm(deconv, scope=sc+'/dc_batch_norm')
        return bn

    def controlable_resblock(net,sc="crb"):
        with tf.variable_scope(sc):
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            net = deconv2d(net,256,3,2,sc = sc + '/deconv_l1')
            net = deconv2d(net,128,3,2,sc = sc + '/deconv_l2')
            net = deconv2d(net,64,3,2,sc = sc + '/deconv_l3')
            net = deconv2d(net,16 ,3,2,sc = sc + '/deconv_l4')  
            net = tf.pad(net, [[0, 0], [4, 4], [4, 4], [0, 0]], "REFLECT")
            return net

    with tf.variable_scope(scope) as sc:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                net = instance_norm(net, name='conv_1/batch_norm')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=False, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=False, sc='conv_ds_7')

                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15') 
                net = controlable_resblock(net)
        
                predictions = tf.nn.tanh(_depthwise_separable_conv(net, 1, width_multiplier, sc='conv_ds_15'))

    return predictions


# def mobilenet_arg_scope(weight_decay=0.0):
#   """Defines the default mobilenet argument scope.

#   Args:
#     weight_decay: The weight decay to use for regularizing the model.

#   Returns:
#     An `arg_scope` to use for the MobileNet model.
#   """
#   with slim.arg_scope(
#       [slim.convolution2d, slim.separable_convolution2d],
#       weights_initializer=slim.initializers.xavier_initializer(),
#       biases_initializer=slim.init_ops.zeros_initializer(),
#       weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
#     return sc
