

import numpy as np
import tensorflow as tf
import os
from dataloader import DataLoader
import nn_builder
from Networks.TrainTestHelper import TrainTestHelper
import argparse
from traintest import train
import random
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping





def get_imagenet_prediction(image, hot_vec,  network, loss_func):
    pred = network(image, training=False)
    i = tf.math.argmax(pred[0])
    loss = loss_func(hot_vec, pred)
    return i, np.array(pred[0])[i], loss

def save_predicted_results(test_images, labels, network, paths, loss_func, title, output_path):
    with open(os.path.join(output_path, "{}.txt".format(title)), 'w') as f:
        correct_sum = 0
        for i in range(len(test_images)):
            pred, score, loss = get_imagenet_prediction(test_images[i][np.newaxis, :,:,:], labels[i], network, loss_func)
            f.write("{} {} {} {}\n".format(paths[i], pred, score, loss))
            if int(pred) == int(labels[i]):
                correct_sum += 1
        f.write("correctness {}\n".format(correct_sum/len(test_images)))



def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, default=1000, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument('--ds_path', type=str, required=True)

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=10)
    parser.add_argument('--learning_rate1', default=1e-3, type=float)
    parser.add_argument('--learning_rate2', default=1e-5, type=float)


    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='number of batches')

    parser.add_argument('--restart_dataloader_config', type=str, default=None)
    parser.add_argument('--restart_model_path', type=str, default=None)

    return parser.parse_args()

def main():
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(args.ds_path,
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=123,
                                                                   labels="inferred",
                                                                   label_mode=int,
                                                                   image_size=args.input_size,
                                                                   batch_size=args.batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(args.ds_path,
                                                                 validation_split=0.2,
                                                                 subset="validation",
                                                                 seed=123,
                                                                 labels="inferred",
                                                                 label_mode=int,
                                                                 image_size=args.input_size,
                                                                 batch_size=args.batch_size)



    # dataloader = DataLoader("dataloader", args.train_path, args.val_path, args.test_path, args.cls_num, args.input_size,
    #                         output_path=args.output_path, restart_config_path=args.restart_dataloader_config)
    network = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
    if args.restart_model_path is not None:
        network.load_model(args.restart_model_path)
    network.update_output_path(args.output_path)

    network.freeze_status()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
    hist = network.fit(train_ds, steps_per_epoch=100, validation_data=val_ds, validation_steps=10,
                               epochs=100, callbacks=[checkpoint, early])

    

if __name__ == "__main__":
    main()
