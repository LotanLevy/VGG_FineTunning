

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from augmentationHelper import get_random_augment
import random
import os
import re

SPLIT_FACTOR = "$"

def image_name(image_path):
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).jpg"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)


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


def write_dataset_map(output_path, dataset_name,  paths, labels):
    assert len(paths) == len(labels)
    with open(os.path.join(output_path, '{}.txt'.format(dataset_name)), 'w') as df:
        lines = ["{}{}{}\n".format(paths[i], SPLIT_FACTOR, labels[i]) for i in range(len(paths))]
        df.writelines(lines)






class DataLoader:

    def __init__(self, name, train_file, val_file, test_file, cls_num, input_size,
                 output_path=os.getcwd(), restart_config_path=None):
        self.classes_num = cls_num
        self.input_size = input_size

        self.name = name
        self.output_path = output_path
        # self.paths_logger = {"train": [], "val": [], "test": []}
        # self.labels_logger = {"train": [], "val": [], "test": []}
        #
        # self.datasets = {"train": read_dataset_map(train_file),
        #                "val": read_dataset_map(val_file),
        #                "test": read_dataset_map(test_file)}



        # self.batches_idx = {"train": 0, "val": 0, "test": 0}

        if restart_config_path is not None:
            args = self.parse_config_file(restart_config_path)
            self.build_by_dataset_settings(args, False)
        else:
            args = [("train", train_file, 0), ("val", val_file,  0), ("test", test_file, 0)]
            self.build_by_dataset_settings(args, True)

        unique_labels = np.unique(self.datasets["train"][1])
        new_labels = np.arange(0, len(unique_labels))
        self.labels_map = dict(zip(unique_labels, new_labels))


    def build_by_dataset_settings(self, datasets_args, to_shaffle):
        self.paths_logger = dict()
        self.labels_logger = dict()
        self.datasets = dict()
        self.batches_idx = dict()
        for args in datasets_args:
            key, dataset_paths_file, batch_idx = args[0], args[1], args[2]
            self.paths_logger[key] = []
            self.labels_logger[key] = []
            self.datasets[key] = read_dataset_map(dataset_paths_file, shuffle=to_shaffle)
            self.batches_idx[key] = batch_idx


    def parse_config_file(self, config_file):
        with open(config_file, 'r') as f:
            lines_list = f.read().splitlines()
            lines_args = [line.split(SPLIT_FACTOR) for line in lines_list]
            parsed_args = [(args[0], args[1], int(args[2])) for args in lines_args]
            return parsed_args



    def write_last_train_dataset_config(self):
        with open(os.path.join(self.output_path, 'last_train_config'), 'w') as df:
            lines = []
            for key in self.datasets:
                dataset_path = os.path.join(self.output_path, '{}.txt'.format(key))
                lines.append("{}{}{}{}{}".format(key, SPLIT_FACTOR, dataset_path, SPLIT_FACTOR, self.batches_idx[key]))
            df.writelines(lines)

    def get_iterations_for_epoch(self, batch_size):
        train_samples = len(self.datasets["train"][0])
        return int(train_samples/batch_size)


    def read_batch(self, batch_size, mode):
        all_paths, all_labels = self.datasets[mode]

        indices = list(range(self.batches_idx[mode], min(self.batches_idx[mode] + batch_size, len(all_paths))))
        if len(indices) < batch_size:
            self.batches_idx[mode] = 0
            rest = batch_size - len(indices)
            indices += list(range(self.batches_idx[mode], min(self.batches_idx[mode] + rest, len(all_paths))))

        self.batches_idx[mode] += batch_size


        batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        labels = []
        b_idx = 0
        for i in indices:
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
                    f.write("{}{}{}\n".format(self.paths_logger[mode][i], SPLIT_FACTOR, self.labels_logger[mode][i]))
        #
        # new_dir = os.path.join(self.output_path, "last_train_settings")
        # for key in self.datasets:
        #     write_dataset_map(new_dir, key, self.datasets[key][0], self.datasets[key][1])
        #
        # self.write_last_train_dataset_config()








