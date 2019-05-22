# DCNN implement of TensorFlow
from __future__ import division
import tensorflow as tf
import numpy as np

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return tf.contrib.slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return tf.contrib.slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def deconv_resnet(image, reuse=False, scope="deconvolution"):
    name = scope
    with tf.variable_scope(name) as scope:
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            y = instance_norm(conv2d(y, dim, 1, s, padding='VALID', name=name+'_c3'), name+'_bn3')
            return y + x

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, 64, 7, 1, padding='VALID', name='d_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, 128, 3, 2, name='d_e2_c'), 'd_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, 256, 3, 2, name='d_e3_c'), 'd_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, 256, name='d_r1')
        r2 = residule_block(r1, 256, name='d_r2')
        r3 = residule_block(r2, 256, name='d_r3')
        r4 = residule_block(r3, 256, name='d_r4')
        r5 = residule_block(r4, 256, name='d_r5')
        r6 = residule_block(r5, 256, name='d_r6')
        r7 = residule_block(r6, 256, name='d_r7')
        r8 = residule_block(r7, 256, name='d_r8')
        r9 = residule_block(r8, 256, name='d_r9')
        r10= residule_block(r9, 256, name='d_r10')
        r11= residule_block(r10, 256, name='d_r11')
        r12= residule_block(r11, 256, name='d_r12')
        r13= residule_block(r12, 256, name='d_r13')

        d1 = deconv2d(r13, 128, 3, 2, name='d_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'd_d1_bn'))
        d2 = deconv2d(d1, 64, 3, 2, name='d_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'd_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, 1, 7, 1, padding='VALID', name='d_pred_c'))
        return pred

def load_initial_weights(session,finetune_layer):
    """Load weights from file into network.
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    come as a dict of lists (e.g. weights['conv1'] is a list) and not as
    dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
    'biases') we need a special load function
    """
    # Load the weights into memory
    WEIGHTS_PATH= 'params.npz'
    weights_dict = np.load(WEIGHTS_PATH)['arr_0'].item()
    # weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()


    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if layer should be trained from scratch
            if op_name not in finetune_layer:
                # Assign weights/biases to their corresponding tf variable

                op = op_name.split('/')[-1]
                with tf.variable_scope(op_name.split('/'+op)[0], reuse=True):

                    var = tf.get_variable(op, trainable=False)
                    session.run(var.assign(weights_dict[op_name]))