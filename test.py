import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
import keras.utils.np_utils as np_utils
import tempfile
import numpy as np
import math

def weight_variable(shape):
        weights = tf.get_variable("weights", shape, initializer=tf.contrib.keras.initializers.he_normal())
        return weights

def bias_variable(shape):
        biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.0))
        return biases

def conv_layer(x, w, b, name, strides, padding='SAME'):
        with tf.variable_scope(name):
            w = weight_variable(w)
            b = bias_variable(b)
            conv_and_biased = tf.nn.conv2d(x, w, strides=strides, padding=padding, name=name) + b
        return conv_and_biased

def batch_normalization(inputs, scope, is_training=True, need_relu=True):
        bn = tf.contrib.layers.batch_norm(inputs,
                                          decay=0.999,
                                          center=True,
                                          scale=True,  # 可以让学生实验True和False的区别。
                                          epsilon=0.001,
                                          activation_fn=None,
                                          param_initializers=None,
                                          param_regularizers=None,
                                          updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
                                          is_training=is_training,  # 可以让学生实验True和False的区别。
                                          reuse=None,
                                          variables_collections=None,
                                          outputs_collections=None,
                                          trainable=True,
                                          batch_weights=None,
                                          fused=False,
                                          data_format='NHWC',
                                          zero_debias_moving_mean=False,
                                          scope=scope,
                                          renorm=False,
                                          renorm_clipping=None,
                                          renorm_decay=0.99)
        if need_relu:
            bn = tf.nn.relu(bn, name='relu')
        return bn

def maxpooling(x, kernal_size, strides, name):  # 最大池化，前面的是核大小，一般为[1, 2, 2, 1]，后面的strides指的是步长，如[1, 2, 2, 1]。
        return tf.nn.max_pool(x, ksize=kernal_size, strides=strides, padding='SAME', name=name)

def avg_pool(input_feats, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(input_feats, ksize, strides, padding)  # 平均池化
        return output

def conv_block(input_tensor, kernel_size, filters, stage, block, train_flag, stride2=False):
        nb_filter1, nb_filter2, nb_filter3 = filters  # nb_filter1/2/3分别是64/64/256，三个整数。
        conv_name_base = 'res' + str(stage) + block + '_branch'  # 'res2a_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'  # 'bn2a_branch'
        _, _, _, input_layers = input_tensor.get_shape().as_list()
        if stride2 == False:
            stride_for_first_conv = [1, 1, 1, 1]  # 。
        else:
            stride_for_first_conv = [1, 2, 2, 1]
        x = conv_layer(input_tensor, [1, 1, input_layers, nb_filter1], [nb_filter1], strides=stride_for_first_conv,
                       padding='SAME', name=conv_name_base + '2a')
        x = batch_normalization(x, scope=bn_name_base + '2a', is_training=train_flag)
        x = conv_layer(x, [kernel_size, kernel_size, nb_filter1, nb_filter2], [nb_filter2], strides=[1, 1, 1, 1],
                       padding='SAME', name=conv_name_base + '2b')
        x = batch_normalization(x, scope=bn_name_base + '2b', is_training=train_flag)
        x = conv_layer(x, [1, 1, nb_filter2, nb_filter3], [nb_filter3], strides=[1, 1, 1, 1], padding='SAME',
                       name=conv_name_base + '2c')
        x = batch_normalization(x, scope=bn_name_base + '2c', is_training=train_flag, need_relu=False)
        shortcut = conv_layer(input_tensor, [1, 1, input_layers, nb_filter3], [nb_filter3],
                              strides=stride_for_first_conv, padding='SAME', name=conv_name_base + '1')
        shortcut = batch_normalization(shortcut, scope=bn_name_base + '1', is_training=train_flag)
        x = tf.add(x, shortcut)
        x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
        return x

def identity_block(input_tensor, kernel_size, filters, stage, block, train_flag):
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        _, _, _, input_layers = input_tensor.get_shape().as_list()
        x = conv_layer(input_tensor, [1, 1, input_layers, nb_filter1], [nb_filter1], strides=[1, 1, 1, 1],
                       padding='SAME', name=conv_name_base + '2a')
        x = batch_normalization(x, scope=bn_name_base + '2a', is_training=train_flag)
        x = conv_layer(x, [kernel_size, kernel_size, nb_filter1, nb_filter2], [nb_filter2], strides=[1, 1, 1, 1],
                       padding='SAME', name=conv_name_base + '2b')
        x = batch_normalization(x, scope=bn_name_base + '2b', is_training=train_flag)
        x = conv_layer(x, [1, 1, nb_filter2, nb_filter3], [nb_filter3], strides=[1, 1, 1, 1], padding='SAME',
                       name=conv_name_base + '2c')
        x = batch_normalization(x, scope=bn_name_base + '2c', is_training=train_flag, need_relu=False)
        x = tf.add(x, input_tensor)
        x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
        return x

def resnet_graph(input_image, architecture, train_flag=bool(1), stage5=False):  # 残差网络函数
        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])  # 上下左右各补3个0。
        x = tf.pad(input_image, paddings, "CONSTANT")
        w = weight_variable([7, 7, 3, 64])
        b = bias_variable([64])  #
        x = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='VALID', name='conv_1') + b
        x = batch_normalization(x, scope='bn_conv1', is_training=train_flag, need_relu=True)
        C1 = x = maxpooling(x, [1, 3, 3, 1], [1, 2, 2, 1], name='stage1')
        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', train_flag=train_flag)  # 结构块。
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_flag=train_flag)
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_flag=train_flag)
        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_flag=train_flag, stride2=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_flag=train_flag)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_flag=train_flag)
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_flag=train_flag)
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_flag=train_flag, stride2=True)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]  # block_count=22，即，下句for循环的range是0~22，正好是加23个层。
        for i in range(block_count):  # 加22个层，都是用identity_block函数加。
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_flag=train_flag)
        C4 = x  #
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_flag=train_flag, stride2=True)
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_flag=train_flag)
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_flag=train_flag)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]


def cost(logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def train(X_train, Y_train,test_features, test_labels):
        features = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.float32, [None, 100])

        [_, _, _, C4, C5] = resnet_graph(features,architecture="resnet50")

        if C5==None:
            C4 = avg_pool(C4, 2)
            flatten = tf.layers.flatten(C4)
            #x = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        else:
            avg_pool(C5, 1)
            flatten = tf.layers.flatten(C5)
            #x = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            #x = tf.nn.dropout(x, keep_prob)
        logits = tf.layers.dense(flatten, units=100)

        train_mode = tf.placeholder(tf.bool, name='training')
        cross_entropy = cost(logits, labels)           #交叉熵来计算损失函数和代价函数
        correction_prediction = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        with tf.name_scope('adam_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        '''
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())
        '''
        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for j in range(10):
                print('epoch %d:'%j)
                for i in range(len(mini_batches)):
                    X_mini_batch, Y_mini_batch = mini_batches[i]
                    train_step.run(
                        feed_dict={features: X_mini_batch, labels: Y_mini_batch, keep_prob: 0.5, train_mode: True})
                    if i % 10 == 0:
                        train_cost = sess.run(cross_entropy, feed_dict={features: X_mini_batch,labels: Y_mini_batch, keep_prob: 1.0,
                                                train_mode: False})
                        acc = correction_prediction.eval(feed_dict={features: X_mini_batch,
                                          labels: Y_mini_batch, keep_prob: 1.0, train_mode: False})
                        print('step %d, training cost %g, accuracy %g' % (i, train_cost, acc))
                test_accuracy = correction_prediction.eval(feed_dict={features: test_features,
                                          labels: test_labels, train_mode: False})
                print('test accuracy %g'% test_accuracy)


def main(_):
    (x_Train, y_Train), (x_Test, y_Test) = cifar100.load_data()
    x_Train4D = x_Train.reshape(x_Train.shape[0], 32, 32, 3).astype('float32')
    x_Test4D = x_Test.reshape(x_Test.shape[0], 32, 32, 3).astype('float32')
    # 归一化
    x_Train4D_normalize = x_Train4D / 255
    x_Test4D_normalize = x_Test4D / 255
    # one-hot Encoding
    y_TrainOneHot = np_utils.to_categorical(y_Train).astype('float32')
    y_TestOneHot = np_utils.to_categorical(y_Test).astype('float32')

    train(x_Train4D_normalize, y_TrainOneHot,x_Test4D_normalize, y_TestOneHot)

    # 显示各类图像
    '''
        fig, ax = plt.subplots(
            nrows=10,
            ncols=10,
            sharex=True,
            sharey=True, )
        ax = ax.flatten()
        for i in range(100):
            img = x_Train[y_Train.reshape(-1) == i][0]
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        #plt.show()
        '''

if __name__ == "__main__":
    tf.app.run(main=main)

