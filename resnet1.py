import tensorflow as tf
import  os
import numpy as np
import pickle

# 文件存放目录
CIFAR_DIR = "./cifar-10-batches-py"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_data(filename):
    '''read data from data file'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes') # python3 需要添加上encoding='bytes'
        return data[b'data'], data[b'labels'] # 并且 在 key 前需要加上 b

class CifarData:
    def __init__(self, filenames, need_shuffle):
        '''参数1:文件夹 参数2:是否需要随机打乱'''
        all_data = []
        all_labels = []

        for filename in filenames:
            # 将所有的数据,标签分别存放在两个list中
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        # 将列表 组成 一个numpy类型的矩阵!!!!
        self._data = np.vstack(all_data)
        # 对数据进行归一化, 尺度固定在 [-1, 1] 之间
        self._data = self._data / 127.5 - 1
        # 将列表,变成一个 numpy 数组
        self._labels = np.hstack(all_labels)
        # 记录当前的样本 数量
        self._num_examples = self._data.shape[0]
        # 保存是否需要随机打乱
        self._need_shuffle = need_shuffle
        # 样本的起始点
        self._indicator = 0
        # 判断是否需要打乱
        if self._need_shuffle:
            self._shffle_data()

    def _shffle_data(self):
        # np.random.permutation() 从 0 到 参数,随机打乱
        p = np.random.permutation(self._num_examples)
        # 保存 已经打乱 顺序的数据
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        '''return batch_size example as a batch'''
        # 开始点 + 数量 = 结束点
        end_indictor = self._indicator + batch_size
        # 如果结束点大于样本数量
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                # 重新打乱
                self._shffle_data()
                # 开始点归零,从头再来
                self._indicator = 0
                # 重新指定 结束点. 和上面的那一句,说白了就是重新开始
                end_indictor = batch_size # 其实就是 0 + batch_size, 把 0 省略了
            else:
                raise Exception("have no more examples")
        # 再次查看是否 超出边界了
        if end_indictor > self._num_examples:
            raise Exception("batch size is larger than all example")

        # 把 batch 区间 的data和label保存,并最后return
        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels

# 拿到所有文件名称
train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
# 拿到标签
test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]

# 拿到训练数据和测试数据
train_data = CifarData(train_filename, True)
test_data = CifarData(test_filename, False)

def residual_block(x, output_channel):
    '''
    定义残差块儿
    :param x: 输入tensor
    :param output_channel: 输出的通道数
    :return: tensor
    需要注意的是:每经过一个stage,通道数就要 * 2
    在同一个stage中,通道数是没有变化的
    '''
    input_channel = x.get_shape().as_list()[-1] # 拿出 输入 tensor 的 最后一维:也就是通道数
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2) #
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("input channel can't match output channel")


    conv1 = tf.layers.conv2d(x,
                              output_channel,
                              (3, 3),
                              strides = strides,
                              padding = 'same',
                              activation = tf.nn.relu,
                              name = 'conv1'
                             )
    conv2 = tf.layers.conv2d(conv1,
                              output_channel,
                              (3, 3),
                              strides = (1, 1), # 因为 上一层 卷积已经进行过降采样,故这里不需要
                              padding = 'same',
                              activation = tf.nn.relu,
                              name = 'conv2'
                             )

    if increase_dim: # 需要使用降采样
        # pooled_x 数据格式 [ None, image_width, image_height, channel ]
        # 要求格式 [ None, image_width, image_height, channel * 2 ]
        pooled_x = tf.layers.average_pooling2d(x,
                                                (2, 2), # size
                                                (2, 2), # stride
                                                padding = 'valid'
                                               )
        '''
        如果输出通道数是输入的两倍的话,需要增加通道数量.
        maxpooling 只能降采样,而不能增加通道数,
        所以需要单独增加通道数
        '''
        padded_x = tf.pad(pooled_x, # 参数 2 ,在每一个通道上 加 pad
                           [
                               [ 0, 0 ],
                               [ 0, 0 ],
                               [ 0, 0 ],
                               [input_channel // 2, input_channel // 2] # 实际上就是 2倍input_channel,需要均分开
                            ]
                          )
    else:
        padded_x = x

    output_x = conv2 + padded_x   # 就是 公式: H(x) = F(x) + x
    return  output_x

def res_net(x, num_residual_blocks, num_filter_base, class_num):
    '''
    残差网络主程序
    :param x:  输入tensor
    :param num_residual_blocks: 每一个stage有多少残差块儿 eg: list [3, 4, 6, 2] 及每一个stage上的残差块儿数量
    :param num_filter_base:  最初的通道数
    :param class_num: 所需要的分类数
    :return: tensor
    '''
    num_subsampling = len(num_residual_blocks) # num_subsampling 为 stage 个数
    layers = [] # 保存每一个残差块的输出
    # x: [ None, width, height, channel] -> [width, height, channel]
    input_size = x.get_shape().as_list()[1:]

    # 首先,开始第一个卷积层
    with tf.variable_scope('conv0'):
        conv0= tf.layers.conv2d(x,
                               num_filter_base,
                               (3, 3),
                               strides = (1, 1),
                                padding = 'same',
                               name = 'conv0'
                               )
        layers.append(conv0)
        # 根据 模型,此处应有一个 pooling,但是 cifar-10 数据集很小,所以不再做 pool
        # num_subsampling = 4, sample_id = [0, 1, 2, 3]
        for sample_id in range(num_subsampling):
            for i in range(num_residual_blocks[sample_id]):
                with tf.variable_scope('conv%d_%d' % (sample_id, i)):
                    conv = residual_block(layers[-1],
                                          num_filter_base * ( 2 ** sample_id ) # 每一个stage都是之前的2倍
                                          )
                    layers.append(conv)


        # 最后就到了 average pool, 1000维 全连接, 这一步
        with tf.variable_scope('fc'):
            # layer[-1].shape: [None, width, height, channel]
            # kernal_size = image_width, image_height
            global_pool = tf.reduce_mean(layers[-1], [1, 2]) # 求平均值函数,参数二 指定 axis
            # global_pool的shape是(?, 128)
            # 这里需要解释一下,对第二维,第三维求平均值,实际上就是对每一个feature map求一个平均值,一共有128个特征图.
            # 所以维度从四维,降低到了两维
            logits = tf.layers.dense(global_pool, class_num)
            layers.append(logits)
        return  layers[-1]


# 设计计算图
# 形状 [None, 3072] 3072 是 样本的维数, None 代表位置的样本数量
x = tf.placeholder(tf.float32, [None, 3072])
# 形状 [None] y的数量和x的样本数是对应的
y = tf.placeholder(tf.int64, [None])

# [None, ], eg: [0, 5, 6, 3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 将最开始的向量式的图片,转为真实的图片类型
x_image = tf.transpose(x_image, perm= [0, 2, 3, 1])


y_ = res_net(x_image, [2, 3, 2], 32, 10)


# 使用交叉熵 设置损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)
# 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy

# 预测值 获得的是 每一行上 最大值的 索引.注意:tf.argmax()的用法,其实和 np.argmax() 一样的
predict = tf.argmax(y_, 1)
# 将布尔值转化为int类型,也就是 0 或者 1, 然后再和真实值进行比较. tf.equal() 返回值是布尔类型
correct_prediction = tf.equal(predict, y)
# 比如说第一行最大值索引是6,说明是第六个分类.而y正好也是6,说明预测正确



# 将上句的布尔类型 转化为 浮点类型,然后进行求平均值,实际上就是求出了准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'): # tf.name_scope() 定义该变量的命名空间
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 将 损失函数 降到 最低

# 初始化变量
init = tf.global_variables_initializer()

batch_size = 2000
batch_size1=20
train_steps = 1000
test_steps = 10
with tf.Session() as sess:
    sess.run(init) # 注意: 这一步必须要有!!
    # 开始训练
    for i in range(train_steps):
        # 得到batch
        batch_data, batch_labels = train_data.next_batch(batch_size)
        # 获得 损失值, 准确率
        loss_val, acc_val, _ = sess.run([loss, accuracy, train_op], feed_dict={x:batch_data, y:batch_labels})
        print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i+1, loss_val, acc_val))
        test_data = CifarData(test_filename, False)
        all_test_acc_val = []
        for j in range(test_steps):
            test_batch_data, test_batch_labels = test_data.next_batch(batch_size1)
            test_acc_val = sess.run([accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels })
            all_test_acc_val.append(test_acc_val)
        test_acc = np.mean(all_test_acc_val)

        print('[Test ] Step: %d, acc: %4.5f' % ((i+1), test_acc))