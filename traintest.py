import matplotlib.pyplot as plt
import os
from dataloader import read_image, image_name
import numpy as np
import tensorflow as tf
import seaborn as sns; sns.set()






def train(epochs, batch_size, trainer, validator, dataloader, print_freq, output_path, model):
    iteration_for_epoch = dataloader.get_iterations_for_epoch(batch_size)
    trainstep = trainer.get_step()
    valstep = validator.get_step()
    logger = TrainLogger(trainer, validator, output_path)
    print(iteration_for_epoch*epochs)
    for i in range(iteration_for_epoch*epochs):
        batch_x, batch_y = dataloader.read_batch(batch_size, "train")
        trainstep(batch_x, batch_y)
        if i % print_freq == 0:
            batch_x, batch_y = dataloader.read_batch(batch_size, "val")
            valstep(batch_x, batch_y)
            logger.update(i)

        if i % iteration_for_epoch == 0:
            model.save_model(int(i), output_path)



def plot_dict(dict, x_key, output_path):
    for key in dict:
        if key != x_key:
            f = plt.figure()
            plt.plot(dict[x_key], dict[key])
            plt.title(key)
            plt.savefig(os.path.join(output_path, key))
            plt.close(f)
    plt.close("all")


class TrainLogger:
    def __init__(self, trainer, validator, output_path):
        self.logs = {"iteration": [], "train_loss": [],  "val_loss": []}
        self.trainer = trainer
        self.validator = validator
        self.output_path = output_path

    def update(self, iteration):
        self.logs["iteration"].append(iteration)
        self.logs["train_loss"].append(float(self.trainer.loss_logger.result()))
        self.logs["val_loss"].append(float(self.validator.loss_logger.result()))
        print("iteration:{} - train loss : {}, val loss : {}".format(iteration,
                                                                     float(self.trainer.loss_logger.result()),
                                                                     float(self.validator.loss_logger.result())))

    def __del__(self):
        plot_dict(self.logs, "iteration", self.output_path)



class HotMapHelper:
    def __init__(self, model, input_size, loss_func):
        self.model = model
        self.input_size = input_size
        self.loss_func = loss_func

    def test_with_square(self, im_path, label, kernel_size, stride, output_path):
        im = read_image(im_path, self.input_size)[np.newaxis, :, :, :]
        dim_r, dim_h = int((im.shape[1] - kernel_size) / stride), int((im.shape[2] - kernel_size) / stride)
        scores = np.zeros((dim_r, dim_h))
        i, j = 0, 0
        r, c = int(np.floor(kernel_size / 2)), int(np.floor(kernel_size / 2))
        while r < im.shape[1] - int(np.ceil(kernel_size / 2)):
            while c < im.shape[2] - int(np.ceil(kernel_size / 2)):
                image_cp = im.copy()
                k1, k2 = int(np.floor(kernel_size / 2)), int(np.ceil(kernel_size / 2))
                image_cp[0, r - k1: r + k2, c - k1: c + k2, :] = 0

                pred = self.model(image_cp)
                loss = self.loss_func(label, pred)

                scores[i, j] = loss
                c += stride
                j += 1
            r += stride
            i += 1
            j = 0
            c = int(np.floor(kernel_size / 2))
        plt.figure()
        ax = sns.heatmap(scores, vmin=np.min(scores), vmax=np.max(scores))
        im_name = image_name(im_path)
        title = "hot_map_of_{}_with_kernel_{}_and_stride_{}".format(im_name, kernel_size, stride)
        plt.title(title)
        plt.savefig(os.path.join(output_path, title + ".png"))

def get_accuracy_and_loss(images_batch, labels, model, loss_func):
    prediction = model(images_batch, training=False)
    loss = loss_func(labels, prediction)
    accuracy = 0
    for i in range(images_batch.shape[0]):
        pred_label = np.argmax(prediction[i])
        if pred_label == labels[i]:
            accuracy += 1
    return loss, accuracy


