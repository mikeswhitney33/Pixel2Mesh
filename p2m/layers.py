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
from p2m.inits import *
import tensorflow as tf

# flags = tf.app.flags
# FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def project(img_feat, x, y, dim):
    x1 = tf.floor(x)
    x2 = tf.minimum(tf.ceil(x), tf.cast(tf.shape(img_feat)[0], tf.float32) - 1)
    y1 = tf.floor(y)
    y2 = tf.minimum(tf.ceil(y), tf.cast(tf.shape(img_feat)[1], tf.float32) - 1)
    Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32)],1))
    Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32)],1))
    Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32)],1))
    Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32)],1))

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
    Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y2,y))
    Q21 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q21)

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y,y1))
    Q12 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q12)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y,y1))
    Q22 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q22)

    outputs = tf.add_n([Q11, Q21, Q12, Q22])
    return outputs

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution(tf.keras.layers.Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, num_supports, support_id, act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.act = act
        self.num_supports = num_supports
        self.support_id = support_id
        self.vars = {}

        for i in range(num_supports):
            self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(i))
        self.vars['bias'] = zeros([output_dim], name='bias')

    def call(self, inputs, supports, **kwargs):
        assert len(supports) == self.num_supports
        support = supports[self.support_id]
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = []
        for i in range(self.num_supports):
            pre_sup = dot(x, self.vars['weights_' + str(i)])
            supports.append(dot(support[i], pre_sup, sparse=True))
        output = tf.add_n(supports)

        # bias
        output += self.vars['bias']

        return self.act(output)

class GraphPooling(tf.keras.layers.Layer):
    """Graph Pooling layer."""
    def __init__(self, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)

        self.pool_id = pool_id

    def call(self, inputs, pool_idx, **kwargs):
        X = inputs

        add_feat = (1/2.0) * tf.reduce_sum(tf.gather(X, pool_idx[self.pool_id-1]), 1)
        outputs = tf.concat([X, add_feat], 0)

        return outputs

class GraphProjection(tf.keras.layers.Layer):
    def call(self, inputs, img_feat, **kwargs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        h = 250 * tf.divide(-Y, -Z) + 112
        w = 250 * tf.divide(X, -Z) + 112

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)

        x = h/(224.0/56)
        y = w/(224.0/56)
        out1 = project(img_feat[0], x, y, 64)

        x = h/(224.0/28)
        y = w/(224.0/28)
        out2 = project(img_feat[1], x, y, 128)

        x = h/(224.0/14)
        y = w/(224.0/14)
        out3 = project(img_feat[2], x, y, 256)

        x = h/(224.0/7)
        y = w/(224.0/7)
        out4 = project(img_feat[3], x, y, 512)
        outputs = tf.concat([coord,out1,out2,out3,out4], 1)
        return outputs
