import matplotlib.pyplot as plt
import os





def train(epochs, batch_size, trainer, validator, dataloader, print_freq, output_path, model):
    max_iteration = epochs * batch_size
    trainstep = trainer.get_step()
    valstep = validator.get_step()
    logger = TrainLogger(trainer, validator, output_path)
    for i in range(max_iteration):
        batch_x, batch_y = dataloader.read_batch(batch_size, "train")
        trainstep(batch_x, batch_y)
        if i % print_freq == 0:
            batch_x, batch_y = dataloader.read_batch(batch_size, "val")
            valstep(batch_x, batch_y)
            logger.update(i)

        if i % epochs == 0:
            model.save_model(int(i/epochs), output_path)



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