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
import tensorflow as tf
from p2m.chamfer import *

def laplace_coord(pred, lape_idx, block_id):
    # lape_idx.shape = [(None, 10), (None, 10), (None, 10)]
    vertex = tf.concat([pred, tf.zeros([1,3])], 0)
    # vertex.shape = (None, 3)
    indices = lape_idx[block_id-1][:, :8]
    # indices.shape = (None, 8)
    weights = tf.cast(lape_idx[block_id-1][:,-1], tf.float32)
    # weight.shape = (None, 1)

    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1,1]), [1,3])
    # weights.shape = (None, 3)
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    # pre-reduce_sum = (8, 3)
    # post-reduce_sum = (8,)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    #  pred - laplace * weights
    #  (8, 3) - (8,) * (8, 3)
    return laplace

def laplace_loss(pred1, pred2, lape_idx, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, lape_idx, block_id)
    lap2 = laplace_coord(pred2, lape_idx, block_id)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1,lap2)), 1)) * 1500

    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    move_loss = tf.cond(tf.equal(block_id,1), lambda:0., lambda:move_loss)
    return laplace_loss + move_loss

def unit(tensor):
    return tf.nn.l2_normalize(tensor, axis=1)

def mesh_loss(pred, labels, edges, block_id):
    gt_pt = labels[:, :3] # gt points
    gt_nm = labels[:, 3:] # gt normals

    # edge in graph
    nod1 = tf.gather(pred, edges[block_id-1][:,0])
    nod2 = tf.gather(pred, edges[block_id-1][:,1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300

    # chamer distance
    dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
    point_loss = (tf.reduce_mean(dist1) + 0.55*tf.reduce_mean(dist2)) * 3000

    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, edges[block_id-1][:,0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    # cosine = tf.where(tf.greater(cosine,0.866), tf.zeros_like(cosine), cosine) # truncated
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss
