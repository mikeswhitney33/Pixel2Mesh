#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import division
# import tflearn
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential

from p2m.layers import *
from p2m.losses import *

# flags = tf.app.flags
# FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, name=None, logging=False):
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        #with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        self.activations.append(self.inputs)
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)

        self.output1 = self.activations[15]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)

        self.output2 = self.activations[31]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)

        self.output3 = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "Data/checkpoint/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "Data/checkpoint/%s.ckpt" % self.name
        #save_path = "checks/tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, learning_rate, weight_decay, feat_dim, coord_dim, hidden, support1, support2, support3, dropout, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.weight_decay = weight_decay
        self.feat_dim = feat_dim
        self.coord_dim = coord_dim
        self.hidden = hidden
        self.support1 = support1
        self.support2 = support2
        self.support3 = support3
        self.dropout = dropout

        self.build()

    def _loss(self):
        '''
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        '''
        self.loss += mesh_loss(self.output1, self.placeholders, 1)
        self.loss += mesh_loss(self.output2, self.placeholders, 2)
        self.loss += mesh_loss(self.output3, self.placeholders, 3)
        self.loss += .1*laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        self.loss += laplace_loss(self.output1_2, self.output2, self.placeholders, 2)
        self.loss += laplace_loss(self.output2_2, self.output3, self.placeholders, 3)

        # Weight decay loss
        conv_layers = range(1,15) + range(17,31) + range(33,48)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)

    def _build(self):
        self.build_cnn18() #update image feature
        # first project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=self.feat_dim,
                                            output_dim=self.hidden,
                                            support=self.support1,
                                            dropout_p=self.dropout))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.hidden,
                                                support=self.support1,
                                                dropout_p=self.dropout))
        self.layers.append(GraphConvolution(input_dim=self.hidden,
                                            output_dim=self.coord_dim,
                                            act=lambda x: x,
                                            support=self.support,
                                            dropout_p=self.dropout))
        # second project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=1)) # unpooling
        self.layers.append(GraphConvolution(input_dim=self.feat_dim+FLAGS.hidden,
                                            output_dim=self.hidden,
                                            support=self.support2,
                                            dropout_p=self.dropout))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.hidden,
                                                support=self.support2,
                                                dropout_p=self.dropout))
        self.layers.append(GraphConvolution(input_dim=self.hidden,
                                            output_dim=self.coord_dim,
                                            act=lambda x: x,
                                            support=self.support2,
                                            dropout_p=self.dropout))
        # third project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=2)) # unpooling
        self.layers.append(GraphConvolution(input_dim=self.feat_dim+self.hidden,
                                            output_dim=self.hidden,
                                            support=self.support3,
                                            dropout_p=self.dropout))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.hidden,
                                                support=self.support3,
                                                dropout_p=self.dropout))
        self.layers.append(GraphConvolution(input_dim=self.hidden,
                                            output_dim=int(self.hidden/2),
                                            support=self.support3,
                                            dropout_p=self.dropout))
        self.layers.append(GraphConvolution(input_dim=int(self.hidden/2),
                                            output_dim=self.coord_dim,
                                            act=lambda x: x,
                                            support=self.support3,
                                            dropout_p=self.dropout))

#     def build_cnn18(self):
#         x=self.placeholders['img_inp']
#         x=tf.expand_dims(x, 0)
# #224 224
#         x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x0=x
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #112 112
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x1=x
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #56 56
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x2=x
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #28 28
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x3=x
#         x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #14 14
#         x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x4=x
#         x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #7 7
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x5=x
# #updata image feature
#         self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})
#         self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3



# tf.keras.layers.Conv2D(
#     filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
#     dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform', bias_initializer='zeros',
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, **kwargs
# )

class GCNModel(tf.keras.Model):
    # args.learning_rate, args.weight_decay, args.feat_dim, args.coord_dim, args.hidden
    def __init__(self, feat_dim, coord_dim, hidden, num_supports):
        super().__init__()
        self.cnn18 = CNN18()
        self.gcn_layers = []

        self.gcn_layers.append(GraphProjection())
        self.gcn_layers.append(GraphConvolution(input_dim=feat_dim,
                                            output_dim=hidden,
                                            num_supports=num_supports,
                                            support_id=1))
        for _ in range(12):
            self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                                output_dim=hidden,
                                                num_supports=num_supports,
                                            support_id=1))
        self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                            output_dim=coord_dim,
                                            act=lambda x: x,
                                            num_supports=num_supports,
                                            support_id=1))
        # second project block
        self.gcn_layers.append(GraphProjection())
        self.gcn_layers.append(GraphPooling(pool_id=1)) # unpooling
        self.gcn_layers.append(GraphConvolution(input_dim=feat_dim + hidden,
                                            output_dim=hidden,
                                            num_supports=num_supports,
                                            support_id=2))
        for _ in range(12):
            self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                                output_dim=hidden,
                                                num_supports=num_supports,
                                            support_id=2))
        self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                            output_dim=coord_dim,
                                            act=lambda x: x,
                                            num_supports=num_supports,
                                            support_id=2))
        # third project block
        self.gcn_layers.append(GraphProjection())
        self.gcn_layers.append(GraphPooling(pool_id=2)) # unpooling
        self.gcn_layers.append(GraphConvolution(input_dim=feat_dim + hidden,
                                            output_dim=hidden,
                                            num_supports=num_supports,
                                            support_id=3))
        for _ in range(12):
            self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                                output_dim=hidden,
                                                num_supports=num_supports,
                                                support_id=3))
        self.gcn_layers.append(GraphConvolution(input_dim=hidden,
                                            output_dim=int(hidden/2),
                                            num_supports=num_supports,
                                            support_id=3))
        self.gcn_layers.append(GraphConvolution(input_dim=int(hidden/2),
                                            output_dim=coord_dim,
                                            act=lambda x: x,
                                            num_supports=num_supports,
                                            support_id=3))

        self.unpool_layer1 = GraphPooling(pool_id=1)
        self.unpool_layer2 = GraphPooling(pool_id=2)

    def call(self, inputs, supports, pool_idx):
        img_feat = self.cnn18(inputs)

        args = {
            "img_feat": img_feat,
            "supports": supports,
            "pool_idx": pool_idx
        }
        # Build sequential resnet model
        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        activations = []
        activations.append(inputs)
        for idx, layer in enumerate(self.gcn_layers):
            hidden = layer(activations[-1], **args)
            if idx in eltwise:
                hidden = tf.add(hidden, activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, activations[-2]], 1)
            activations.append(hidden)

        output1 = activations[15]
        output1_2 = self.unpool_layer1(output1)

        output2 = activations[31]
        output2_2 = self.unpool_layer2(self.output2)

        output3 = activations[-1]

        return output1, output2, output3





def conv2(features, kernel_size, stride):
    return Conv2D(features, kernel_size, stride, activation="relu", kernel_regularizer=l2(1e-5), padding='same')

def conv_block(features, n, first_k=3):
    conv = [conv2(features, first_k, 2)]
    conv.extend([conv2(features, 3, 1) for _ in range(n-1)])
    return Sequential(conv)

class CNN18(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv16 = conv_block(16, 2)
        self.conv32 = conv_block(32, 2)  # Sequential([conv2(32, 3, 2), conv2(32, 3, 1)])
        self.conv64 = conv_block(64, 3)  # Sequential([conv2(64, 3, 2), conv2(64, 3, 1), conv2(64, 3, 1)])
        self.conv128 = conv_block(128, 3)  # Sequential([conv2(128, 3, 2), conv2(128, 3, 1), conv2(128, 3, 1)])
        self.conv256 = conv_block(256, 3, 5)
        self.conv512 = conv_block(512, 3, 5)

    def call(self, x):
        x = tf.expand_dims(x, 0)
        x0 = self.conv16(x)
        x1 = self.conv32(x0)
        x2 = self.conv64(x1)
        x3 = self.conv128(x2)
        x4 = self.conv256(x3)
        x5 = self.conv512(x4)

        ret = [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]
        return ret

#     def build_cnn18(self):
#         x=self.placeholders['img_inp']
#         x=tf.expand_dims(x, 0)
# #224 224
#         x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x0=x
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #112 112
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x1=x
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #56 56
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x2=x
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #28 28
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x3=x
#         x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #14 14
#         x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x4=x
#         x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
# #7 7
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
#         x5=x
# #updata image feature
#         self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})
#         self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3


