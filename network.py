import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def load(data_path, session):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

def get_unique_name(prefix):        
    id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
    return '%s_%d' % (prefix, id)

# def make_var_w(name, shape):
#     return tf.get_variable(name, shape, regularizer= tf.contrib.layers.l2_regularizer)

def make_var_b(name, shape):
    return tf.get_variable(name, shape)

def make_var_w(name, shape):
    return tf.get_variable(name, shape, regularizer= tf.contrib.layers.l2_regularizer(scope ='weights',scale=0.05))

def conv(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1):
    c_i = int(input.get_shape()[-1])
    # print(c_i)
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_var_w('weights', shape=[k_h, k_w, c_i/group, c_o])
        biases = make_var_b('biases', [c_o])
        if group == 1:
            conv = convolve(input, kernel)
        else:
            # WARNING with Tensorflow 0.12 the order of args sor  "Split" has changed
            # input_groups = tf.split(3, group, input)
            # kernel_groups = tf.split(3, group, kernel)
            input_groups = tf.split(input, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            # conv = tf.concat(3, output_groups)
            conv = tf.concat(output_groups, 3) # update for FT 0.12

        if relu:
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            return tf.nn.relu(bias, name=scope.name)
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)

def relu(input, name):
    return tf.nn.relu(input, name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

def concat(inputs, axis, name):
    return tf.concat(concat_dim=axis, values=inputs, name=name)

def fc(input, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = make_var_w('weights', shape=[num_in, num_out])
        biases = make_var_b('biases', [num_out])
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(input, weights, biases, name=scope.name)
        return fc

def softmax(input, name):
    return tf.nn.softmax(input, name)

def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

