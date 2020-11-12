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
import tensorflow as tf
from p2m.layers import *
from p2m.losses import *


def conv_2d(filters, kernel_size, strides):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides, 'same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(1e-5))

class CNN18(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_2d(16, (3, 3), 1)
        self.conv2 = conv_2d(16, (3, 3), 1)
        self.conv3 = conv_2d(32, (3, 3), 2)
        self.conv4 = conv_2d(32, (3, 3), 1)
        self.conv5 = conv_2d(32, (3, 3), 1)
        self.conv6 = conv_2d(64, (3, 3), 2)
        self.conv7 = conv_2d(64, (3, 3), 1)
        self.conv8 = conv_2d(64, (3, 3), 1)
        self.conv9 = conv_2d(128, (3, 3), 2)
        self.conv10 = conv_2d(128, (3, 3), 1)
        self.conv11 = conv_2d(128, (3, 3), 1)
        self.conv12 = conv_2d(256, (5, 5), 2)
        self.conv13 = conv_2d(256, (3, 3), 1)
        self.conv14 = conv_2d(256, (3, 3), 1)
        self.conv15 = conv_2d(512, (5, 5), 2)
        self.conv16 = conv_2d(512, (3, 3), 1)
        self.conv17 = conv_2d(512, (3, 3), 1)
        self.conv18 = conv_2d(512, (3, 3), 1)

    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs, 0)
        x = self.conv1(x)
        x = self.conv2(x)
        x0 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x1 = x
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x2 = x
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x3 = x
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x4 = x
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x5 = x
        return x0, x1, x2, x3, x4, x5


class Pix2Mesh(tf.keras.Model):
    def __init__(self, feat_dim, hidden, coord_dim):
        super().__init__()
        self.cnn18 = CNN18()
        self.glayers = []
        self.glayers.append(GraphProjection())
        self.glayers.append(GraphConvolution(input_dim=feat_dim, output_dim=hidden, support_idx=0))
        for _ in range(12):
            self.glayers.append(GraphConvolution(input_dim=hidden, output_dim=hidden, support_idx=0))
        self.glayers.append(GraphConvolution(input_dim=hidden, output_dim=coord_dim, act=lambda x: x,  support_idx=0))
        # second project block
        self.glayers.append(GraphProjection())
        self.glayers.append(GraphPooling(pool_id=1)) # unpooling
        self.glayers.append(GraphConvolution(input_dim=feat_dim+hidden,
                                            output_dim=hidden,
                                            support_idx=1))
        for _ in range(12):
            self.glayers.append(GraphConvolution(input_dim=hidden,
                                                output_dim=hidden,
                                                support_idx=1))
        self.glayers.append(GraphConvolution(input_dim=hidden,
                                            output_dim=coord_dim,
                                            act=lambda x: x,
                                            support_idx=1))
        # third project block
        self.glayers.append(GraphProjection())
        self.glayers.append(GraphPooling(pool_id=2)) # unpooling
        self.glayers.append(GraphConvolution(input_dim=feat_dim+hidden,
                                            output_dim=hidden,
                                            support_idx=2))
        for _ in range(12):
            self.glayers.append(GraphConvolution(input_dim=hidden,
                                                output_dim=hidden,
                                                support_idx=2))
        self.glayers.append(GraphConvolution(input_dim=hidden,
                                            output_dim=int(hidden/2),
                                            support_idx=2))
        self.glayers.append(GraphConvolution(input_dim=int(hidden/2),
                                            output_dim=coord_dim,
                                            act=lambda x: x,
                                            support_idx=2))

        self.unpool1 = GraphPooling(pool_id=1)
        self.unpool2 = GraphPooling(pool_id=2)

    def call(self, inputs, training=False):
        img_inp, features, supports, pool_idx = inputs
        _, _, x2, x3, x4, x5 = self.cnn18(img_inp)
        img_feat = [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]

        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        activations = []
        activations.append(features)
        for idx, layer in enumerate(self.glayers):
            if isinstance(layer, GraphProjection):
                hidden = layer((activations[-1], img_feat), training=training)
            elif isinstance(layer, GraphConvolution):
                hidden = layer((activations[-1], supports), training=training)
            elif isinstance(layer, GraphPooling):
                hidden = layer((activations[-1], pool_idx), training=training)
            else:
                hidden = layer(activations[-1], training=training)
            if idx in eltwise:
                hidden = tf.add(hidden, activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, activations[-2]], 1)
            activations.append(hidden)

        output1 = activations[15]
        output1_2 = self.unpool1((output1, pool_idx), training=training)

        output2 = activations[31]
        output2_2 = self.unpool2((output2, pool_idx), training=training)

        output3 = activations[-1]
        return output1, output1_2, output2, output2_2, output3
