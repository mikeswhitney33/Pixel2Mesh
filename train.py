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
import argparse
import pickle
import numpy as np
import tensorflow as tf
from p2m.utils import construct_feed_dict
from p2m.models import GCNModel
from p2m.fetcher import DataFetcher
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set random seed
seed = 1024
np.random.seed(seed)
# tf.compat.v1.set_random_seed(seed)
tf.random.set_seed(seed)

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--data_list", type=str, default="Data/train_list.txt", help="Data list.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
parser.add_argument("--hidden", type=int, default=256, help="Number of units in hidden layer.")
parser.add_argument("--feat_dim", type=int, default=963, help="Number of units in feature layer.")
parser.add_argument("--coord_dim", type=int, default=3, help="Number of units in output layer.")
parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for L2 loss.")
args = parser.parse_args()
# flags = tf.compat.v1.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_list', 'Data/train_list.txt', 'Data list.') # training data list
# flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
# flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
# flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.') # gcn hidden layer channel
# flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') # image feature dim
# flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
# flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

# Define placeholders(dict) and model
# num_blocks = 3
num_supports = 2
# placeholders = {
#     'features': tf.compat.v1.placeholder(tf.float32, shape=(None, 3)),
#     'img_inp': tf.compat.v1.placeholder(tf.float32, shape=(224, 224, 3)),
#     'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, 6)),
#     'support1': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'support2': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'support3': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'faces': [tf.compat.v1.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  #for face loss, not used.
#     'edges': [tf.compat.v1.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
#     'lape_idx': [tf.compat.v1.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], #for laplace term
#     'pool_idx': [tf.compat.v1.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] #for unpooling
# }
#  def __init__(self, placeholders, learning_rate, weight_decay, feat_dim, coord_dim, hidden, **kwargs):
# model = GCN(placeholders, args.learning_rate, args.weight_decay, args.feat_dim, args.coord_dim, args.hidden , logging=True)
# def __init__(self, feat_dim, coord_dim, hidden, num_supports):
model = GCNModel(args.feat_dim, args.coord_dim, args.hidden, num_supports)


# Load data, initialize session
data = DataFetcher(args.data_list)
data.setDaemon(True) ####
data.start()
# config=tf.compat.v1.ConfigProto()
# #config.gpu_options.allow_growth=True
# config.allow_soft_placement=True
# sess = tf.compat.v1.Session(config=config)
# sess.run(tf.compat.v1.global_variables_initializer())
#model.load(sess)

@tf.function
def train_step(features, img_inp, labels, support1, support2, support3, faces, edges, lape_idx, pool_idx):
    out1, out2, out3, rloss = model(img_inp, [support1, support2, support3], pool_idx)

    # def call(self, inputs, supports, pool_idx):
    return None, rloss, out1, out2, out3

# Train graph model
train_loss = open('record_train_loss.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(args.learning_rate))
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'), encoding='latin')
feed_dict = construct_feed_dict(pkl)

train_number = data.number
for epoch in range(args.epochs):
    all_loss = np.zeros(train_number,dtype='float32')
    for iters in range(train_number):
        # Fetch training data
        img_inp, y_train, data_id = data.fetch()
        feed_dict.update({placeholders['img_inp']: img_inp})
        feed_dict.update({placeholders['labels']: y_train})

        # Training step
        _, dists,out1,out2,out3 = train_step(**feed_dict)#  sess.run([model.opt_op,model.loss,model.output1,model.output2,model.output3], feed_dict=feed_dict)
        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        if (iters+1) % 128 == 0:

            print(f"Epoch {epoch+1}, Iteration {iters + 1}")
            print(f"Mean loss = {mean_loss}, iter_loss = {dists}, {data.queue.qsize()}")
    # Save model
    # model.save(sess)
    train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
    train_loss.flush()

data.shutdown()
print("Training finished")
