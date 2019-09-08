from keras.callbacks import Callback


class YpredCallback(Callback):
    """Used as a callback for Keras to get Ypred_train, and Ypred_test for each epoch.

    Example:
    yc = YpredCallback(X, X)
    model.fit(X, Y, callbacks=[yc]
    """

    def __init__(self, model, X_train, X_test=None):
        self.model = model  # Keras model
        self.Y_train = []
        self.Y_test = []
        self.X_train = X_train
        # If X_test is None, use X_train
        if X_test is None:
            self.X_test = X_train
        else:
            self.X_test = X_test

    def on_epoch_end(self, model, epoch, logs=None):
        Y_train_pred = self.model.predict(self.X_train).flatten()
        Y_test_pred = self.model.predict(self.X_test).flatten()
        self.Y_train.append(Y_train_pred)
        self.Y_test.append(Y_test_pred)
