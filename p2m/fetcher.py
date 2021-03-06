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
import os
import pickle
import queue as Queue
import sys
import threading

import numpy as np
import tensorflow as tf
from skimage import io, transform


class FeedDict:
    def __init__(self, pkl):
        self.features = pkl[0]
        self.edges = [pkl[i][1][0] for i in range(1, 4)]
        self.faces = pkl[5]
        self.pool_idx = pkl[4]
        self.lape_idx = pkl[7]
        support1 = [tf.cast(tf.sparse.SparseTensor(*pkl[1][i]), tf.float32) for i in range(len(pkl[1]))]
        support2 = [tf.cast(tf.sparse.SparseTensor(*pkl[2][i]), tf.float32) for i in range(len(pkl[2]))]
        support3 = [tf.cast(tf.sparse.SparseTensor(*pkl[3][i]), tf.float32) for i in range(len(pkl[3]))]
        self.supports = (support1, support2, support3)


class DataFetcher(threading.Thread):
    def __init__(self, file_list):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = Queue.Queue(64)

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        pkl_path = self.pkl_list[idx]
        label = pickle.load(open(pkl_path, 'rb'), encoding="latin")

        img_path = pkl_path.replace('.dat', '.png')
        '''
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img[np.where(img[:,:,3]==0)] = 255
        img = cv2.resize(img, (224,224))
        img = img[:,:,:3]/255.
        '''
        img = io.imread(img_path)
        img[np.where(img[:,:,3]==0)] = 255
        img = transform.resize(img, (224,224))
        img = img[:,:,:3].astype('float32')

        pkl_path2 = pkl_path.replace("rendering", "features")
        with open(pkl_path2, "rb") as file:
            pkl = pickle.load(file, encoding="latin")
        feed = FeedDict(pkl)

        return img, label, pkl_path.split('/')[-1], feed

    def run(self):
        while self.index < 90000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':
    file_list = sys.argv[1]
    data = DataFetcher(file_list)
    data.start()

    image,point,normal,_,_ = data.fetch()
    print(image.shape)
    print(point.shape)
    print(normal.shape)
    data.stopped = True
