import tensorflow as tf


class TrainTestHelper:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, loss_func, training=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_logger = tf.keras.metrics.Mean(name='loss')
        self.training = training



    def get_step(self):

        @tf.function()
        def train_step(inputs, labels):
            with tf.GradientTape(persistent=True) as tape:

                # Descriptiveness loss
                prediction = self.model(inputs, training=self.training)
                loss_value = self.loss_func(labels, prediction)
                self.loss_logger(loss_value)

            if self.training:
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return train_step


