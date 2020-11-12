import argparse
import os

import tensorflow as tf
import tqdm
import numpy as np

from p2m_utils import path_utils
from p2m.losses import mesh_loss, laplace_loss
from p2m.models import Pix2Mesh
from p2m.fetcher import DataFetcher


def loss_fn(model, inputs, labels, loss_args, training=False):
    _, features, _, _ = inputs
    output1, output1_2, output2, output2_2, output3 = model(inputs, training)
    edges, lape_idx, weight_decay = loss_args
    loss = 0
    loss += mesh_loss(output1, labels, edges, 1)
    loss += mesh_loss(output2, labels, edges, 2)
    loss += mesh_loss(output3, labels, edges, 3)
    loss += 0.1 * laplace_loss(features, output1, lape_idx, 1)
    loss += laplace_loss(output1_2, output2, lape_idx, 2)
    loss += laplace_loss(output2_2, output3, lape_idx, 3)

    conv_layers = list(range(1,15)) + list(range(17,31)) + list(range(33,48))
    for layer_id in conv_layers:
        for var in model.layers[layer_id].trainable_variables:
            loss += weight_decay * tf.nn.l2_loss(var)
    return loss


def grad(model, *args, **kwargs):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, *args, **kwargs)
    return loss, tape.gradient(loss, model.trainable_variables)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", "-wd", type=float, default=5e-6)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--max-batches", "-mb", type=int, default=-1)
    parser.add_argument("--outfile", "-o", type=str, default=os.path.join(path_utils.get_data_dir(), "checkpoint"))
    args = parser.parse_args()

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    max_batches = args.max_batches
    outfile = args.outfile

    feat_dim = 963
    hidden = 256
    coord_dim = 3
    data_list = os.path.join(path_utils.get_data_dir(), "train_list.txt")

    model = Pix2Mesh(feat_dim, hidden, coord_dim)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function(experimental_relax_shapes=True)
    def train_step(inputs, labels, loss_args):
        nonlocal model, optimizer
        loss, grads = grad(model, inputs, labels, loss_args, training=True)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    data = DataFetcher(data_list)
    data.setDaemon(True) ####
    data.start()

    train_number = min(data.number, data.number if max_batches < 0 else max_batches)

    epoch_pbar = tqdm.tqdm(range(epochs))
    for _ in epoch_pbar:
        all_loss = np.zeros(train_number, dtype='float32')
        mean_loss = 0
        batch_pbar = tqdm.tqdm(range(train_number), leave=False)
        for iters in batch_pbar:
            img_inp, y_train, _, feed = data.fetch()

            inputs = img_inp, feed.features, feed.supports, feed.pool_idx
            loss_args = feed.edges, feed.lape_idx, weight_decay

            loss = train_step(inputs, y_train, loss_args)
            all_loss[iters] = loss
            mean_loss = np.mean(all_loss[np.where(all_loss)])
            if (iters+1) % 128 == 0:
                batch_pbar.desc = f'Mean loss = {mean_loss}, iter loss = {loss}, {data.queue.qsize()}'
        epoch_pbar.desc = f"Epoch Loss = {mean_loss}"
        model.save(outfile)
    data.shutdown()
    print('Training Finished!')


if __name__ == "__main__":
    main()
