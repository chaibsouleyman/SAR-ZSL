# -*- coding: utf-8 -*-
"""
@author: Cherry
"""

from __future__ import division
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io as sio

types = 10
test_size = 446
Batch_Size = 50
Epoches = 6
img_size = 112
is_training = False
st = time.time()

def load_test_data():
    filedir = 'data/mstar_test.npz'
    data = np.load(filedir)
    data_test = data['data_test']
    label_test = data['label_test']
    return data_test, label_test

def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(input, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    vgg_rawnet = sio.loadmat('data/imagenet-vgg-verydeep-19-SQ.mat')
    vgg_layers = vgg_rawnet['layers'][0]
    conv1_1 = build_net('conv', input, get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
    conv1_2 = build_net('conv', conv1_1, get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
    pool1 = build_net('pool', conv1_2)
    conv2_1 = build_net('conv', pool1, get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
    conv2_2 = build_net('conv', conv2_1, get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
    pool2 = build_net('pool', conv2_2)
    conv3_1 = build_net('conv', pool2, get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
    conv3_2 = build_net('conv', conv3_1, get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
    conv3_3 = build_net('conv', conv3_2, get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
    conv3_4 = build_net('conv', conv3_3, get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
    pool3 = build_net('pool', conv3_4)
    conv4_1 = build_net('conv', pool3, get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
    conv4_2 = build_net('conv', conv4_1, get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
    conv4_3 = build_net('conv', conv4_2, get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
    conv4_4 = build_net('conv', conv4_3, get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
    return conv4_4


def fn(input):
    input = tf.reshape(input, [-1, 14 * 14 * 512])
    fc1 = slim.fully_connected(input, 2048, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_1')
    fc2 = slim.fully_connected(fc1, 1024, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_2')
    fc3 = slim.fully_connected(fc2, 512, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_3')
    fc4 = slim.fully_connected(fc3, 256, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_4')
    fc5 = slim.fully_connected(fc4, 128, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_5')
    fc6 = slim.fully_connected(fc5, 64, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_6')
    pred = slim.fully_connected(fc6, types, activation_fn=tf.nn.softmax, scope='fc_7')
    return pred


sess = tf.Session()
with tf.variable_scope(tf.get_variable_scope()):
    label = tf.placeholder(tf.float32, [None, types])
    input_image = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
    vgg_output = build_vgg19(input_image)
    predict = fn(vgg_output)
    G_loss = -tf.reduce_mean(label * tf.log(predict + 0.00001))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1)), "float"))

lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=[var for var in tf.trainable_variables() if
                                                                            var.name.startswith('fc_')])
saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

# load data

if is_training:
    # load data
    data_train, label_train, data_test, label_test = load_data()
    data_train.shape = -1, img_size, img_size, 1
    data_test.shape = -1, img_size, img_size, 1
    size_of_train = len(data_train)
    batch_idxs = size_of_train // Batch_Size
    loss_batch = np.zeros(batch_idxs * Epoches, dtype=float)
    acc_batch = np.zeros(batch_idxs * Epoches, dtype=float)
    loss_epoch = np.zeros([Epoches])
    loss_T72_epoch = np.zeros([Epoches])
    loss_test_epoch = np.zeros([Epoches])
    acc_T72_epoch = np.zeros([Epoches])
    acc_test_epoch = np.zeros([Epoches])

    cnt = 0
    for epoch in range(Epoches):
        idx = np.random.permutation(size_of_train)
        g_loss = np.zeros([batch_idxs])
        for ind in range(batch_idxs):

            batch_data = data_train[idx[ind*Batch_Size:(ind+1)*Batch_Size]]
            batch_data.shape = -1,img_size,img_size,1
            batch_label = label_train[idx[ind*Batch_Size:(ind+1)*Batch_Size]]
            _,G_current = sess.run([G_opt,G_loss],feed_dict={label:batch_label,input_image:batch_data,lr: 1e-5})
            g_loss[ind] = G_current
            loss_batch[cnt] = G_current
            acc_batch[cnt] = sess.run(acc,feed_dict={label:batch_label,input_image:batch_data})
            cnt+=1
            print("Epoch:%d || Batch:%d || Mean Loss:%.4f || Acc:%.4f || Cost Time:%.2f"%(epoch, ind, np.mean(g_loss[np.where(g_loss)]),acc_batch[cnt-1],time.time()-st))

        saver.save(sess, "checkpoint/model.ckpt")

else:
    # load data
    data_test, label_test = load_test_data()

    # load saved model
    saver.restore(sess, "checkpoint/model.ckpt")
    test_pred = np.zeros([len(data_test), types])
    for ind in range(len(data_test)//Batch_Size):
        batch_data = data_test[ind*Batch_Size:(ind+1)*Batch_Size]
        batch_data.shape = -1, img_size, img_size, 1
        test_pred[ind*Batch_Size:(ind+1)*Batch_Size] = sess.run(predict, feed_dict={input_image: batch_data})
    batch_data = data_test[(ind+1)*Batch_Size:]
    batch_data.shape = -1,img_size,img_size,1
    test_pred[(ind+1)*Batch_Size:] = sess.run(predict, feed_dict={input_image: batch_data})
    sio.savemat('./result/pred.mat', {'label_test': label_test, 'label_pred': test_pred})
