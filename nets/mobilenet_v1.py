# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import numpy as np

'''
import keras
from keras.layers.wrappers import TimeDistributed
from keras.layers import AveragePooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Lambda#'''

from collections import namedtuple

from nets.network import Network
from model.config import cfg

def separable_conv2d_same(inputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D separable convolution with 'SAME' padding.
  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.
  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """

  # By passing filters=None
  # separable_conv2d produces only a depth-wise convolution layer
  if stride == 1:
    return slim.separable_conv2d(inputs, None, kernel_size, 
                                  depth_multiplier=1, stride=1, rate=rate,
                                  padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.separable_conv2d(inputs, None, kernel_size, 
                                  depth_multiplier=1, stride=stride, rate=rate, 
                                  padding='VALID', scope=scope)

# The following is adapted from:
# https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py

# Conv and DepthSepConv named tuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(kernel=3, stride=1, depth=64),
    DepthSepConv(kernel=3, stride=2, depth=128),
    DepthSepConv(kernel=3, stride=1, depth=128),
    DepthSepConv(kernel=3, stride=2, depth=256),
    DepthSepConv(kernel=3, stride=1, depth=256),
    DepthSepConv(kernel=3, stride=2, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    # use stride 1 for the 13th layer
    DepthSepConv(kernel=3, stride=1, depth=1024),
    DepthSepConv(kernel=3, stride=1, depth=1024)
]

# Modified mobilenet_v1
def mobilenet_v1_base(inputs,
                      conv_defs,
                      starting_layer=0,
                      min_depth=8,
                      depth_multiplier=1.0,
                      output_stride=None,
                      reuse=None,
                      scope=None):
  """Mobilenet v1.
  Constructs a Mobilenet v1 network from inputs to the given final endpoint.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    starting_layer: specifies the current starting layer. For region proposal 
      network it is 0, for region classification it is 12 by default.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef named tuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. 
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
  Raises:
    ValueError: if depth_multiplier <= 0, or convolution type is not defined.
  """
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse):
    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    for i, conv_def in enumerate(conv_defs):
      end_point_base = 'Conv2d_%d' % (i + starting_layer)

      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= conv_def.stride
      else:
        layer_stride = conv_def.stride
        layer_rate = 1
        current_stride *= conv_def.stride

      if isinstance(conv_def, Conv):
        end_point = end_point_base
        net = resnet_utils.conv2d_same(net, depth(conv_def.depth), conv_def.kernel,
                          stride=conv_def.stride,
                          scope=end_point)

      elif isinstance(conv_def, DepthSepConv):
        end_point = end_point_base + '_depthwise'
        
        net = separable_conv2d_same(net, conv_def.kernel,
                                    stride=layer_stride,
                                    rate=layer_rate,
                                    scope=end_point)

        end_point = end_point_base + '_pointwise'

        net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                          stride=1,
                          scope=end_point)

      else:
        raise ValueError('Unknown convolution type %s for layer %d'
                         % (conv_def.ltype, i))

    return net

# Modified arg_scope to incorporate configs
def mobilenet_v1_arg_scope(is_training=True,
                           stddev=0.09):
  batch_norm_params = {
      'is_training': False,
      'center': True,
      'scale': True,
      'decay': 0.9997,
      'epsilon': 0.001,
      'trainable': False,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(cfg.MOBILENET.WEIGHT_DECAY)
  if cfg.MOBILENET.REGU_DEPTH:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None

  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      trainable=is_training,
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, 
                      normalizer_fn=slim.batch_norm,
                      padding='SAME'):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

class mobilenetv1(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
    self._scope = 'MobilenetV1'

  def _image_to_head(self, is_training, reuse=None):
    # Base bottleneck
    assert (0 <= cfg.MOBILENET.FIXED_LAYERS <= 12)
    net_conv = self._image
    if cfg.MOBILENET.FIXED_LAYERS > 0:
      with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        net_conv = mobilenet_v1_base(net_conv,
                                      _CONV_DEFS[:cfg.MOBILENET.FIXED_LAYERS],
                                      starting_layer=0,
                                      depth_multiplier=self._depth_multiplier,
                                      reuse=reuse,
                                      scope=self._scope)

    if cfg.MOBILENET.FIXED_LAYERS < 12:
      with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        net_conv = mobilenet_v1_base(net_conv,
                                      _CONV_DEFS[cfg.MOBILENET.FIXED_LAYERS:7],
                                      starting_layer=cfg.MOBILENET.FIXED_LAYERS,
                                      depth_multiplier=self._depth_multiplier,
                                      reuse=reuse,
                                      scope=self._scope)

        net_conv_dw4 = net_conv
        net_conv = mobilenet_v1_base(net_conv,
                                      _CONV_DEFS[7:12],
                                      starting_layer=7,
                                      depth_multiplier=self._depth_multiplier,
                                      reuse=reuse,
                                      scope=self._scope)


    self._act_summaries.append(net_conv)
    self._layers['head'] = net_conv

    return net_conv, net_conv_dw4

  def _head_to_tail(self, pool5, is_training, reuse=None):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
      fc7 = mobilenet_v1_base(pool5,
                              _CONV_DEFS[12:],
                              starting_layer=12,
                              depth_multiplier=self._depth_multiplier,
                              reuse=reuse,
                              scope=self._scope)
      # average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
    return fc7


  def _WSPP(self, pool5, is_training, reuse=None):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
      fc7 = mobilenet_v1_base(pool5,
                              _CONV_DEFS[12:],
                              starting_layer=12,
                              depth_multiplier=self._depth_multiplier,
                              reuse=reuse,
                              scope=self._scope)#特征维度为256*7*7*1024  = 256*50176
      # WSPP  to learn scale-invariance information
      fc7_flatten = slim.flatten(fc7, scope='fc7_flatten')
      fc7_flatten = tf.expand_dims(fc7_flatten, 2)
      fc7_flatten = tf.expand_dims(fc7_flatten, 3)
      scale_1 = slim.max_pool2d(fc7_flatten, [25, 1], stride = [25, 1], padding='SAME', scope='scale_1')
      #print('11111111111111111111111111',scale_1)# 256 644
      scale_2 = slim.max_pool2d(fc7_flatten, [51, 1], stride = [51, 1], padding='SAME', scope='scale_2')
      #print('11111111111111111111111111',scale_2)# 256 202
      scale_3 = slim.max_pool2d(fc7_flatten, [102, 1], stride = [102, 1], padding='SAME', scope='scale_3')
      #print('11111111111111111111111111',scale_3)# 256 101
      scale_4 = slim.max_pool2d(fc7_flatten, [204, 1], stride = [204, 1], padding='SAME', scope='scale_4')
      #print('11111111111111111111111111',scale_4)# 256 51           '''
      #scale_5 = slim.max_pool2d(fc7_flatten, [2000, 1], stride = [2000, 1], padding='SAME', scope='scale_5')
      #print('11111111111111111111111111',scale_5)# 256 26           '''
      scale = tf.concat([scale_1, scale_2], 1)
      scale = tf.concat([scale, scale_3], 1)
      scale = tf.concat([scale, scale_4], 1)
      #scale = tf.concat([scale, scale_5], 1)
      scale = tf.squeeze(scale, 3)
      scale = tf.squeeze(scale, 2)
      scale = slim.fully_connected(scale, 128, scope='scale_reduce')
      #print('11111111111111111111111111', scale)  # 256 1024
      #fc7 = tf.reduce_mean(fc7, axis=[1, 2])     # 17
    return scale


  '''
  def _CLSTM_atten(self, pool5, is_training):
    #print(pool5) #Tensor("vgg_16/transpose:0", shape=(1, 256, 7, 7, 512), dtype=float32)
    batch_size, feature_h, feature_w, out_channels = pool5.get_shape().as_list()
    pool5_temp = tf.expand_dims(pool5,0)
    #pool5_atten = ConvLSTM2D(out_channels, kernel_size=(3, 3), padding='same', return_sequences=True, name='ROI_attd3')(pool5_temp)# 1 256 7 7 512
    pool5_atten = ConvLSTM2D(1, kernel_size=(1, 1), padding='same', return_sequences=True, name='ROI_attd1')(pool5_temp)
    #pool5_atten = TimeDistributed(AveragePooling2D(pool_size=(feature_h,feature_w),  strides=(feature_h, feature_w), padding='same'))(pool5_atten)
    pool5_atten = tf.nn.softmax(pool5_atten, name='softmax_ROI_att')
    pool5_atten = tf.tile(pool5_atten, [1, 1, 1, 1, out_channels])
    pool5_atten = tf.squeeze(pool5_atten, 0)
    pool5_atten = tf.reshape(pool5_atten, [-1,feature_h, feature_w, out_channels])
    #pool5_atten = Lambda(lambda pool5_atten: keras.backend.reshape(pool5_atten, (batch_size,feature_h, feature_w, out_channels)))(pool5_atten)
    pool5 = pool5 * pool5_atten
    return pool5#'''

  def _CLSTM_atten(self, fc7, fc7_conv_dw4, is_training):
    #print(pool5) #Tensor("vgg_16/transpose:0", shape=(1, 256, 7, 7, 512), dtype=float32)
    fc7_temp = tf.expand_dims(fc7, 1)
    fc7_conv_dw4_temp = tf.expand_dims(fc7_conv_dw4, 1)
    fc7_temp = tf.concat([fc7_temp, fc7_conv_dw4_temp], 1)
    batch_size, timestep, out_channels = fc7_temp.get_shape().as_list()
    fc7_temp = tf.unstack(fc7_temp, timestep, 1)
    #seq_len = tf.fill([batch_size], time)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=out_channels)
    cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=out_channels)

    output, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=cell, cell_bw=cell1, inputs=fc7_temp, dtype=tf.float32)
    #print('*****************',output)
    #print(output_state_fw.h)
    #print(output_state_bw.h)

    #pool5_atten1 = tf.nn.softmax(output_state_fw.h, name='softmax_ROI_att1')
    #pool5_atten2 = tf.nn.softmax(output_state_bw.h, name='softmax_ROI_att2')#'''
    #output_atten = tf.nn.softmax(output, name='softmax_ROI_att2')#'''
    '''
    lstm_last_state_fw = lstm_last_state[0]
    lstm_last_state_bw = lstm_last_state[1]
    pool5_atten1 = tf.nn.softmax(lstm_last_state_fw, name='softmax_ROI_att1')
    pool5_atten2 = tf.nn.softmax(lstm_last_state_bw, name='softmax_ROI_att2')#'''

    #fc7_new = fc7 + output_state_bw.h
    #fc7_new = tf.concat([fc7, output_state_bw.h], 1)

    #print('*******************',fc7_new)

    return output[1]

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/Conv2d_0/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix MobileNet V1 layers..')
    with tf.variable_scope('Fix_MobileNet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR, and match the scale by (255.0 / 2.0)
        Conv2d_0_rgb = tf.get_variable("Conv2d_0_rgb", 
                                    [3, 3, 3, max(int(32 * self._depth_multiplier), 8)], 
                                    trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/Conv2d_0/weights": Conv2d_0_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv2d_0/weights:0"], 
                           tf.reverse(Conv2d_0_rgb / (255.0 / 2.0), [2])))
