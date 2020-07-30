import argparse
import os
from dataloader import DataLoader
import nn_builder
import tensorflow as tf
import traintest

def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, required=True, help='The number of classes in the dataset')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='A path of the weights')


    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--test_size', '-ts', type=int, default=200, help='number of batches')
    parser.add_argument('--kernel_size', '-ks', type=int, default=24)
    parser.add_argument('--stride', '-s', type=int, default=10)
    return parser.parse_args()


def main():
    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    dataloader = DataLoader("dataloader", args.train_path, args.val_path, args.test_path, args.cls_num, args.input_size,
                            output_path=args.output_path)
    network = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
    network.load_model(args.ckpt_dir)

    counter = 0
    step = 32


    batch_x, batch_y = dataloader.read_batch(step, "test")
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    loss, accuracy = traintest.get_accuracy_and_loss(batch_x, batch_y, network, loss_func)
    print("loss: {}, accuracy: {}".format(loss, accuracy))

    counter += step

    hot_map_creator = traintest.HotMapHelper(network, args.input_size, loss_func)
    paths = dataloader.paths_logger
    labels = dataloader.labels_logger


    for i in range(len(paths)):
        hot_map_creator.test_with_square(paths["test"][i], labels["test"][i], args.kernel_size, args.stride, args.output_path)




if __name__ == "__main__":
    main()
