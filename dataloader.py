

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from augmentationHelper import get_random_augment

import os

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

    def __init__(self, name, train_file, val_file, test_file, cls_num, input_size,
                 output_path=os.getcwd()):
        self.classes_num = cls_num
        self.input_size = input_size

        self.name = name
        self.output_path = output_path
        self.paths_logger = {"train": [], "val": [], "test": []}
        self.labels_logger = {"train": [], "val": [], "test": []}

        self.datasets = {"train": read_dataset_map(train_file),
                       "val": read_dataset_map(val_file),
                       "test": read_dataset_map(test_file)}

        unique_labels = np.unique(self.datasets["train"][1])
        new_labels = np.arange(0, len(unique_labels))
        self.labels_map = dict(zip(unique_labels, new_labels))


    def read_batch(self, batch_size, mode):
        all_paths, all_labels = self.datasets[mode]
        rand_idx = np.random.randint(low=0, high=len(all_paths)-1, size=batch_size).astype(np.int)

        batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        labels = []
        b_idx = 0
        for i in rand_idx:
            batch_images[b_idx, :, :, :] = read_image(all_paths[i], self.input_size, augment=(mode == "train"))
            label = self.labels_map[all_labels[i]]
            self.paths_logger[mode].append(all_paths[i])
            self.labels_logger[mode].append(label)
            labels.append(label)
            b_idx += 1
        return batch_images, np.array(labels)

    def __del__(self):
        for mode in self.paths_logger:
            with open(os.path.join(self.output_path, "{}_{}.txt".format(self.name, mode)), 'w') as f:
                for i in range(len(self.paths_logger[mode])):
                    f.write("{}{}{}".format(self.paths_logger[mode][i], SPLIT_FACTOR, self.labels_logger[mode][i]))






