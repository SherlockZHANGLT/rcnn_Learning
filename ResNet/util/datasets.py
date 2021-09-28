#encoding:UTF-8
import numpy as np
'''
@param labels:保留哪些 标签  / which labels will be used
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_minst(labels):
    print("load pics...")
    pic_train=unpickle('data_batch_1')
    for i in range(1,5):
        pic_train=np.concatenate(pic_train,unpickle('data_batch_'+i))
    pic_test=unpickle('test_batch')
    if labels != None:
        data_train = pic_train.data[pic_train.labels==labels[0]]
        data_test = pic_test.data[pic_test.labels==labels[0]]
        for i in range(1,len(labels)):
            data_train = np.concatenate((pic_train.data[pic_train.labels==labels[i]]),axis=0)
            data_test = np.concatenate((data_test,pic_test.data[pic_test.labels==labels[i]]),axis=0)
    X_train = np.array(data_train[:,1:])
    y_train =  np.array(data_train[:,0])
    X_test =  np.array(data_test[:,1:])
    y_test = np.array(data_test[:,0])
    print("data loaded...")
    return X_train,y_train,X_test,y_test
#设置最大迭代次数
    max_steps = 10000
    #设置每次训练的数据大小
    batch_size = 128
    # 设置数据的存放目录
    cifar10_dir = "C:\\Users\\zlj\\Downloads\\cifar-10-batches-bin"
    #训练集
    #distored_inputs函数产生训练需要使用的数据，包括特征和其对应的label,返回已经封装好的tensor，每次执行都会生成一个batch_size的数量的样本
    images_train,labels_train = cifar10_input.distorted_inputs(cifar10_dir,batch_size)
    #测试集
    images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=cifar10_dir
                                                   ,batch_size=batch_size)
    #载入数据
    image_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,24,24,3])
    #裁剪后尺寸为24×24，彩色图像通道数为3
    label_holder = tf.placeholder(dtype=tf.int32,shape=[batch_size])