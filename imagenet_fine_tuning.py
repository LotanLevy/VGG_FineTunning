

import numpy as np
import tensorflow as tf
import os
from dataloader import DataLoader
import utils
from Networks.imagenet_traintest import TrainTestHelper
import argparse





def train(dataloader, trainer, validator, batches, max_iteration, print_freq):
    np.random.seed(1234)
    tf.random.set_seed(1234)

    trainstep = trainer.get_step()
    valstep = validator.get_step()
    train_dict = {"iteration":[], "loss": []}

    for i in range(max_iteration):
        batch_x, batch_y = dataloader.read_batch(batches, "train")
        trainstep(batch_x, batch_y)
        if i % print_freq == 0:  # validation loss
            batch_x, batch_y = dataloader.read_batch(batches, "val")
            valstep(batch_x, batch_y)

            train_dict["iteration"].append(i)
            train_dict["loss"].append(float(validator.loss_logger.result()))
            print("iteration {} - loss {}".format(i + 1, train_dict["loss"][-1]))


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
    parser.add_argument('--nntype', default="PerceptualModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, default=1000, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchs_num', '-bs', type=int, default=2, help='number of batches')
    parser.add_argument('--train_iterations', '-iter', type=int, default=800, help='The maximum iterations for learning')

    return parser.parse_args()

def main():
    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    dataloader = DataLoader(args.train_path, args.val_path, args.test_path, args.cls_num, args.input_size,
                            name="dataloader", output_path=args.output_path)
    network = utils.get_network(args.nntype)
    network.freeze_layers(19)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    trainer = TrainTestHelper(network, optimizer, loss, training=True)
    validator = TrainTestHelper(network, optimizer, loss, training=False)

    test_images, labels = dataloader.read_batch(200, "test")
    save_predicted_results(test_images, labels, network, dataloader.paths_logger["test"], loss, "before_training", args.output_path)


    train(dataloader, trainer, validator, args.batchs_num, args.train_iterations, args.print_freq)
    save_predicted_results(test_images, labels, network, dataloader.paths_logger["test"], loss, "after_training", args.output_path)




if __name__ == "__main__":
    main()
