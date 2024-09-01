import tensorflow as tf
import tensorflow.contrib.slim as slim
from model.config import cfg

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return tf.reduce_mean(x, [1,2])

def Relu(x):
    return tf.nn.relu(x)

def Fully_connected(x, units, is_training, initializer):
    return slim.fully_connected(x, units, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None)

def Squeeze_excitation_layer(input_x, out_dim, ratio, is_training, initializer):
  
    squeeze = Global_Average_Pooling(input_x)

    excitation = Fully_connected(squeeze, int(out_dim / ratio), is_training, initializer)
    excitation = Relu(excitation)
    excitation = Fully_connected(excitation, int(out_dim), is_training, initializer)
    excitation = Sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation

    return scale

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def conv_layer(inpt, filter_shape, stride, is_training, initializer):
    out_channels = filter_shape[3]
    '''
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")#
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)#'''

    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': 'moving_vars',
          'moving_variance': 'moving_vars',}
      }

    out = slim.conv2d(inpt, out_channels, [filter_shape[0], filter_shape[1]],
                          stride=stride, padding='SAME', activation_fn=tf.nn.relu, 
                          normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                          trainable=is_training, weights_initializer=initializer)
    # 参考https://blog.csdn.net/chanbo8205/article/details/87183631

    #out = tf.nn.relu(batch_norm)

    return out


def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list


def res2net_block(inpt, output_depth, slice_num, ratio, is_training, initializer):
    input_depth = inpt.get_shape().as_list()[3]
    conv1 = conv_layer(inpt, [1, 1, input_depth, output_depth], 1, is_training, initializer)
    slice_list = slice_layer(conv1, slice_num, output_depth)
    side = conv_layer(slice_list[1], [3, 3, output_depth//slice_num, output_depth//slice_num], 1, is_training, initializer)
    z = tf.concat([slice_list[0], side], axis=-1)
    for i in range(2, len(slice_list)):
        y = conv_layer(tf.add(side, slice_list[i]), [3, 3, output_depth//slice_num, output_depth//slice_num], 1, is_training, initializer)
        side = y
        z = tf.concat([z, y], axis=-1)
        #print('z', z)
    conv3 = conv_layer(z, [1, 1, output_depth, input_depth], 1, is_training, initializer)
    conv3 = Squeeze_excitation_layer(conv3, input_depth, ratio, is_training, initializer)
    res = conv3 + inpt
    res = Squeeze_excitation_layer(res, input_depth, ratio, is_training, initializer)

    return res
'''
X = tf.placeholder("float", [8, 256, 256, 256])
net = res2net_block(X, 128, 4)
print('net', net)
slice_ = slice_layer(net, 4, 256)
print('slice_list', slice_)#'''
