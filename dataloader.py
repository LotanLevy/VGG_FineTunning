

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import random
import os
from augmentationHelper import get_random_augment


SPLIT_FACTOR = "$"


def read_image(path, resize_image=(), augment=False):
    image = Image.open(path, 'r')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if augment:
        image = get_random_augment(image, resize_image)
    if len(resize_image) > 0:
        image = image.resize(resize_image, Image.NEAREST)
    image = np.array(image).astype(np.float32)
    return image

def read_dataset_map(data_map_path, shuffle=False):
    with open(data_map_path, "r") as lf:
        lines_list = lf.read().splitlines()
        if shuffle:
            random.shuffle(lines_list)
        lines = [line.split(SPLIT_FACTOR) for line in lines_list]
        images, labels = [], []
        if len(lines) > 0:
            images, labels = zip(*lines)
        labels = [int(label) for label in labels]
    return images, np.array(labels).astype(np.int)

class DataLoader:
    def __init__(self, name, dataset_file, cls_num, input_size, output_path, augment=False):
        self.classes_num = cls_num
        self.input_size = input_size
        self.augment = augment

        self.name = name
        self.output_path = output_path
        self.paths_logger = []
        self.labels_logger = []
        self.batch_idx = 0



        self.datasets = read_dataset_map(dataset_file, shuffle=True)
        unique_labels = np.unique(self.datasets[1])

        # assert len(unique_labels) == cls_num
        new_labels = np.arange(0, len(unique_labels))
        self.labels_map = dict(zip(unique_labels, new_labels))
        print(self.labels_map)
        self.batches_idx = 0
        self.epochs = 0




    def read_batch(self, batch_size):
        all_paths, all_labels = self.datasets

        # takes the next batch, if it finish the epoch it'll start  new epoch

        indices = list(range(self.batches_idx, min(self.batches_idx + batch_size, len(all_paths))))
        if len(indices) < batch_size: # new epoch
            self.batches_idx = 0
            rest = batch_size - len(indices)
            indices += list(range(self.batches_idx, min(self.batches_idx + rest, len(all_paths))))
            self.epochs += 1
        self.batches_idx += batch_size

        batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        paths = []
        labels = []
        b_idx = 0
        for i in indices:
            batch_images[b_idx, :, :, :] = read_image(all_paths[i], self.input_size, augment=self.augment)
            paths.append(all_paths[i])
            labels.append(self.labels_map[all_labels[i]])
            self.paths_logger.append(all_paths[i])
            self.paths_logger.append(self.labels_map[all_labels[i]])
            b_idx += 1

        hot_vecs = tf.keras.utils.to_categorical(np.array(labels), num_classes=self.classes_num)
        return batch_images, np.array(labels)

    # def read_batch(self, batch_size):
    #     batch_images, hot_vecs, paths, labels = self.read_batch_with_details(batch_size)
    #     self.paths_logger += paths
    #     self.labels_logger += labels
    #     return batch_images, hot_vecs

    def __del__(self):
        with open(os.path.join(self.output_path, "{}.txt".format(self.name)), 'w') as f:
            for i in range(len(self.paths_logger)):
                f.write("{}{}{}\n".format(self.paths_logger[i], SPLIT_FACTOR, self.labels_logger[i]))

