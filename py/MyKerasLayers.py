from __future__ import print_function
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('1.')
import os
import sys
import keras
assert keras.__version__.startswith('2.')
from keras.engine import Layer, InputSpec
from keras import activations, regularizers, constraints
from keras import initializers
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from keras.utils import conv_utils
from keras.utils.conv_utils import conv_output_length
from keras.activations import tanh, linear


class AcrossChannelLRN(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (1 + \frac{\alpha}{n} \sum_j x_j^2 )^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from Lasagne, which is from pylearn2.
    This layer is time consuming. Without this layer, it takes 4 sec for 100 iterations, with this layer, it takes 8 sec.
    """

    def __init__(self, local_size=5, alpha=1e-4, beta=0.75, k=1,
                 **kwargs):
        super(AcrossChannelLRN, self).__init__(**kwargs)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        if self.local_size % 2 == 0:
            print("Only works with odd local_size!!!")

    def build(self, input_shape):
        print('No trainable weights for LRN layer.')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return tf.nn.lrn(x, depth_radius=self.local_size, bias=self.k, alpha=self.alpha, beta=self.beta, name=self.name)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "local_size": self.local_size,
                  "alpha": self.alpha,
                  "beta": self.beta,
                  "k": self.k}
        base_config = super(AcrossChannelLRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# alias
LRN_across_channel = AcrossChannelLRN

class XLayer_b(Layer):
    '''
    modified from Convolution2D layer
    '''

    def __init__(self, filters,
                 kernel_size,
                 weight_param_net,  # only one more input argument
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        print("Using XLayer_b layer.")
        super(XLayer_b, self).__init__(**kwargs)
        # from _Conv
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        # make sure my settings overwrite the existing ones
        assert self.data_format == 'channels_last'
        self.weight_param_net = weight_param_net
        self.uses_learning_phase = self.weight_param_net.uses_learning_phase
        # print("weight_param_net's uses_learning_phase: {}".format(self.weight_param_net.uses_learning_phase))
        # print("XLayer's uses_learning_phase: {}".format(self.uses_learning_phase))
        assert self.padding in {'valid', 'same'}, 'padding must be in {valid, same}'
        assert self.data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        assert self.dilation_rate == (1, 1)
        self.nb_row = self.kernel_size[1]  # it will be used later
        self.nb_col = self.kernel_size[0]
        if not self.use_bias:
            print('NOTE: without use_bias term')

    def build(self, input_shape):
        # Input shape
        # 4D tensor with shape:
        # `(samples, channels, rows, cols)` if data_format='channels_first'
        # or 4D tensor with shape:
        # `(samples, rows, cols, channels)` if data_format='channels_last'.
        if self.data_format == 'channels_first':
            print("Only support tf order: channel_last")
            sys.exit(1)
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.trainable_weights = self.weight_param_net.trainable_weights
        self.non_trainable_weights = self.weight_param_net.non_trainable_weights

        # from Convolution2D layer
        if self.data_format == 'channels_first':
            print("Only support tf order: channel_last")
            sys.exit(1)
            kernel_shape = (self.filters, input_dim) + self.kernel_size
        elif self.data_format == 'channels_last':
            kernel_shape = self.kernel_size + (input_dim, self.filters)  # H, W, in_chan, out_chan
        else:
            raise ValueError('Invalid data_format:', self.data_format)
        if not hasattr(self, 'batch_input_shape'):
            # only suitable for fixed batch size because this will only be called when building the model
            self.batch_input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        else:
            assert self.batch_input_shape == input_shape
        print('self.batch_input_shape: {}'.format(self.batch_input_shape))
        print('input_shape: {} x {} x {} x {}'.format(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
        print('Batch size: {}'.format(input_shape[0]))
        print('shape of the weights in XLayer_b {}: {}'.format(self.name, kernel_shape))
        # print('>>>> output shape of weight_param_net: {}'.format(self.weight_param_net.output.get_shape()))
        # print('>>>> weight shape: {}, {}, {}, {}'.format(self.nb_row, self.nb_col, input_dim, self.filters))
        if self.use_bias:
            temp_dim = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            # batch_size never changes
            self.W = [K.reshape(self.weight_param_net.output[idx][:temp_dim], kernel_shape)
                      for idx in range(self.batch_input_shape[0])]
            print('self.W:')
            print('    type: {}'.format(type(self.W)))
            print('    length: {}'.format(len(self.W)))
            print('    shape of self.W[0]: {}'.format(self.W[0]))
            self.b = [self.weight_param_net.output[idx][temp_dim:]
                      for idx in range(self.batch_input_shape[0])]
            # print('>>>> shape of self.b: {}'.format(self.weight_param_net.output[idx][temp_dim:].get_shape()))
            print('self.b:')
            print('    type: {}'.format(type(self.b)))
            print('    length: {}'.format(len(self.b)))
        else:
            self.W = [K.reshape(self.weight_param_net.output[idx], kernel_shape)
                      for idx in range(self.batch_input_shape[0])]

        # this regularizer loss should work on the weights of the sub-network
        self.add_loss(self.weight_param_net.losses)
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, x, mask=None):
        batch_output_shape = self.compute_output_shape(self.batch_input_shape)
        self.split_inputs = [K.expand_dims(x[idx], axis=0) for idx in range(self.batch_input_shape[0])]  # do conv sample by sample

        print('batch size: {}'.format(batch_output_shape[0]))
        for idx in range(batch_output_shape[0]):
            cur_output = K.conv2d(
                self.split_inputs[idx],
                self.W[idx],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
            if self.use_bias:
                if self.data_format == 'channels_first':
                    print("Only support tf order: channel_last")
                    sys.exit(1)
                    cur_output += K.reshape(self.b[idx], (1, self.filters, 1, 1))
                elif self.data_format == 'channels_last':
                    cur_output += K.reshape(self.b[idx], (1, 1, 1, self.filters))
                else:
                    raise ValueError('Invalid data_format:', self.data_format)
            if idx == 0:
                outputs = cur_output
            else:
                outputs = tf.concat([outputs, cur_output], axis=0)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        # change nothing here
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        # change nothing here
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(XLayer_b, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
