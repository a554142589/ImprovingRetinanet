# from keras.layers import Activation, Reshape, Lambda, Multiply, dot, add
# from keras.layers import Conv1D, Conv2D, Conv3D
# from keras.layers import MaxPool1D
# from keras import backend as K
from tensorflow.keras.layers import Activation, Reshape, Lambda, Multiply, dot, add
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras import backend as K


def non_local_block(ip, out_channels, height, width, name, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).

    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.

    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    # channel_dim = -1 here
    # print('##########')
    # print(channel_dim)
    ip_shape = K.shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if ip_shape.shape[0] == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip.get_shape().as_list()

    elif ip_shape.shape[0] == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip.get_shape().as_list()
        else:

            batchsize, dim1, dim2, channels = ip.get_shape().as_list()
            dim1 = height
            dim2 = width
            channels = out_channels

    elif ip_shape.shape[0] == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip.get_shape().as_list()
        else:
            batchsize, dim1, dim2, dim3, channels = ip.get_shape().as_list()

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = out_channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim, name=name+'_dot1')
        theta = Reshape((-1, intermediate_dim))(theta)
        # theta = tf.transpose(theta, [0, 2, 1])
        # phi path
        phi = _convND(ip, rank, intermediate_dim, name=name+'_dot2')
        phi = Reshape((-1, intermediate_dim))(phi)

        # print(phi.shape)
        f = dot([theta, phi], axes=[2,2])
        # f = Activation('softmax')(f)
        # f = Multiply()([theta, phi])
        # print(f.shape)
        # f = tf.nn.softmax(f)
        # print('######DEBUG######')
        # f = f / tf.cast(tf.shape(f)[-1], tf.float32)
        # # print(type(f))
        # # size = K.cast(size,'float32')
        # print(f)
        # # print(size)
        # size = tf.cast(size,'float32')

        # a = tf.to_float(size)
        # scale the values to make it size invariant
        # f = Lambda(lambda z: (1. / (a[-1])) * z)(f)
        # print(size)
        # f = f / size[-1]
        # K.cast()
        # f = f / K.cast(K.shape(f)[-1], 'float32')
        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / K.cast(height*width, 'float32')) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim, name=name+'_g')
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels, name=name+'_y')

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels, name):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', name=name)(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x

#
# def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
#     batchsize, height, width, in_channels = input_x.get_shape().as_list()
#     with tf.variable_scope(scope) as sc:
#         with tf.variable_scope('g') as scope:
#             g = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='g')
#             if sub_sample:
#                 g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')
#
#         with tf.variable_scope('phi') as scope:
#             phi = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='phi')
#             if sub_sample:
#                 phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')
#
#         with tf.variable_scope('theta') as scope:
#             theta = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='theta')
#
#         g_x = tf.reshape(g, [batchsize,out_channels, -1])
#         g_x = tf.transpose(g_x, [0,2,1])
#
#         theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
#         theta_x = tf.transpose(theta_x, [0,2,1])
#         phi_x = tf.reshape(phi, [batchsize, out_channels, -1])
#
#         f = tf.matmul(theta_x, phi_x)
#         # ???
#         f_softmax = tf.nn.softmax(f, -1)
#         y = tf.matmul(f_softmax, g_x)
#         y = tf.reshape(y, [batchsize, height, width, out_channels])
#         with tf.variable_scope('w') as scope:
#             w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
#             if is_bn:
#                 w_y = slim.batch_norm(w_y)
#         z = input_x + w_y
#         return z
#
#
# def weight_variable(shape):
#     initial = tf.contrib.layers.xavier_initializer_conv2d()
#     return tf.Variable(initial(shape=shape))
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.01, shape=shape)
#     return tf.Variable(initial)
#
# def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
#     W = weight_variable([kernel_size, kernel_size, in_features, out_features])
#     conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
#     if with_bias:
#         return conv + bias_variable([out_features])
#     return conv
#
#
# def _non_local_block(input_tensor, computation_compression=2, mode='dot'):
#     if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
#         raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`,`concatenate`')
#
#     input_shape = input_tensor.get_shape().as_list()
#     print(input_shape)
#     batchsize, dim1, dim2, channels = input_shape
#
#     if mode == 'gaussian':  # Gaussian instantiation
#         x1 = tf.reshape(input_tensor, shape=[-1, dim1 * dim2, channels])
#         x2 = tf.reshape(input_tensor, shape=[-1, dim1 * dim2, channels])
#
#         f = tf.matmul(x1, x2, transpose_b=True)
#
#         f = tf.reshape(f, shape=[-1, dim1 * dim2 * dim1 * dim2])
#
#         f = tf.nn.softmax(f, axis=-1)
#
#         f = tf.reshape(f, shape=[-1, dim1 * dim2, dim1 * dim2])
#
#         print("gaussian=", f)
#     elif mode == 'dot':
#         theta = conv2d(input_tensor, channels, channels // 2, 1)  # add BN relu layer before conv will speed up training
#         theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])
#
#         phi = conv2d(input_tensor, channels, channels // 2, 1)
#         phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])
#
#         f = tf.matmul(theta, phi, transpose_b=True)
#
#         # scale the values to make it size invarian t
#         f = f / (dim1 * dim2 * channels)
#
#         print("dot f=", f)
#
#     elif mode == 'concatenate':  # this operation cost too much memory, so make sure you input a small resolution  feature map like(14X14 7X7)
#
#         theta = conv2d(input_tensor, channels, channels // 2, 1)
#         theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])
#
#         phi = conv2d(input_tensor, channels, channels // 2, 1)
#         phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])
#
#         theta_splits = tf.split(theta, dim1 * dim2, 1)
#         phi_splits = tf.split(phi, dim1 * dim2, 1)
#
#         theta_split_shape = tf.shape(theta[0])
#         print("theta_split_shape", theta_split_shape)
#
#         initial = tf.constant(1.0 / channels, shape=[channels, 1])
#
#         print('initial', initial)
#         W_concat = tf.Variable(initial)
#
#         print("W_concat", W_concat)
#
#         f_matrix = []
#         for i in range(dim1 * dim2):
#             for j in range(dim1 * dim2):
#                 print(i, '  ', j)
#                 tmp = tf.concat([theta_splits[i], phi_splits[j]], 2)
#                 tmp = tf.reshape(tmp, shape=[-1, channels])
#                 # print(tmp)
#                 tmp = tf.matmul(tmp, W_concat)
#                 print(tmp)
#                 f_matrix.append(tmp)
#
#         f_matrix_tensor = tf.stack(f_matrix, axis=2)
#         print('f_matrix_tensor', f_matrix_tensor)
#
#         f = tf.reshape(f_matrix_tensor, shape=[-1, dim1 * dim2, dim1 * dim2])
#
#         f = f / (dim1 * dim2 * channels)
#
#         print("concatenate f=", f)
#
#
#     else:  # Embedded Gaussian instantiation
#         theta = conv2d(input_tensor, channels, channels // 2, 1)
#         theta = tf.reshape(theta, shape=[-1, dim1 * dim2, channels // 2])
#
#         phi = conv2d(input_tensor, channels, channels // 2, 1)
#         phi = tf.reshape(phi, shape=[-1, dim1 * dim2, channels // 2])
#
#         if computation_compression > 1:
#             phi = tf.layers.max_pooling1d(phi, pool_size=2, strides=computation_compression, padding='SAME')
#             print('phi', phi)
#
#         f = tf.matmul(theta, phi, transpose_b=True)
#
#         phi_shape = phi.get_shape().as_list()
#         f = tf.reshape(f, shape=[-1, dim1 * dim2 * phi_shape[1]])
#
#         f = tf.nn.softmax(f, axis=-1)
#
#         f = tf.reshape(f, shape=[-1, dim1 * dim2, phi_shape[1]])
#
#         print("Embedded f=", f)
#
#     g = conv2d(input_tensor, channels, channels // 2, 1)
#     g = tf.reshape(g, shape=[-1, dim1 * dim2, channels // 2])
#
#     if computation_compression > 1 and mode == 'embedded':
#         g = tf.layers.max_pooling1d(g, pool_size=2, strides=computation_compression, padding='SAME')
#         print('g', g)
#
#     y = tf.matmul(f, g)
#
#     print('y=', y)
#
#     y = tf.reshape(y, shape=[-1, dim1, dim2, channels // 2])
#
#     y = conv2d(y, channels // 2, channels, kernel_size=3)
#     print('y=', y)
#
#     residual = input_tensor + y
#
#     return residual