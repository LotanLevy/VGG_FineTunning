

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

SPLIT_FACTOR = "$"


def read_image(path, resize_image=()):
    image = Image.open(path, 'r')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if len(resize_image) > 0:
        image = image.resize(resize_image, Image.NEAREST)
    image = np.array(image).astype(np.float32)
    return image

def read_dataset_map(data_map_path):
    with open(data_map_path, "r") as lf:
        lines_list = lf.read().splitlines()
        lines = [line.split(SPLIT_FACTOR) for line in lines_list]
        images, labels = [], []
        if len(lines) > 0:
            images, labels = zip(*lines)
        labels = [int(label) for label in labels]
    return images, np.array(labels).astype(np.int)



class DataLoader:

    def __init__(self, name, dataset_file, cls_num, input_size, output_path=os.getcwd()):
        self.classes_num = cls_num
        self.input_size = input_size

        self.name = name
        self.output_path = output_path
        self.paths_logger = []
        self.labels_logger = []

        self.datasets = read_dataset_map(dataset_file)

        unique_labels = np.unique(self.datasets[1])
        new_labels = np.arange(0, len(unique_labels))
        self.labels_map = dict(zip(unique_labels, new_labels))


    def read_batch(self, batch_size):
        all_paths, all_labels = self.datasets
        rand_idx = np.random.randint(low=0, high=len(all_paths)-1, size=batch_size).astype(np.int)

        batch_labels = all_labels[rand_idx]
        batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        # labels = []
        b_idx = 0
        for i in rand_idx:
            batch_images[b_idx, :, :, :] = read_image(all_paths[i], self.input_size)
            label = self.labels_map[all_labels[i]]
            self.paths_logger.append(all_paths[i])
            self.labels_logger.append(label)
            # labels.append(label)
            b_idx += 1
        return batch_images, batch_labels

    def __del__(self):
        with open(os.path.join(self.output_path, "{}.txt".format(self.name)), 'w') as f:
            for i in range(len(self.paths_logger)):
                f.write("{}{}{}".format(self.paths_logger[i], SPLIT_FACTOR, self.labels_logger[i]))






