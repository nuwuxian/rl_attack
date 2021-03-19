import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
import pdb

class MimicModel(object):

    def __init__(self, input_shape, action_shape):
        self.num_class = action_shape
        self.input_shape = input_shape

        # todo add gpu support
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.session = tf.Session(config=tf_config)
        K.set_session(self.session)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                self.model = self.build_model()
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential([
            Dense(64, input_shape=(13,),name="MnistModel/d1"),
            Activation('relu', name="MnistModel/r1"),
            Dense(32, name="MnistModel/d2"),
            Activation('relu', name="MnistModel/r2"),
            Dense(2, name="MnistModel/d3")
        ])

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

        return model

    def fit(self, x_train, y_train, batch_size, epoch):

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)

        self.model.fit(x = x_train,
                       y = y_train,
                       batch_size = batch_size,
                       validation_split=0.33,
                       epochs = epoch,
                       callbacks=[early_stopping])
        return 0

    def evaluate(self, x, y, batch_size=32, verbose = 0):
        if x.ndim ==3:
            x = np.expand_dims(x, 0)
        loss, acc = self.model.evaluate(x = x, y = y, batch_size= batch_size, verbose = verbose)
        return loss, acc

    def predict(self, x, batch_size=32, verbose = 0):
        if x.ndim ==3:
            x = np.expand_dims(x, 0)
        pred = self.model.predict(x = x, batch_size = batch_size, verbose = verbose)
        return pred

    def predict_for_RL(self, x):
        pred = self.model.predict(x=x, steps=1)
        return pred

    def save(self, model_url):
        self.model.save(model_url)
        return 0

    def load(self, model_url):
        self.model = load_model(model_url)
        return 0



if __name__ == "__main__":


    def load_data(filename):
        import pickle as pkl
        with open(filename, 'rb') as f:
            [state, action] = pkl.load(f)
        return state, action

    #  the filename
    filename = '../saved/trajectory.pkl'
    data_X_true, data_Y_true = load_data(filename)

    from sklearn.model_selection import KFold

    model_candidates = []

    kfold = KFold(n_splits=10, shuffle=True, random_state=1992)
    cvscores = []
    for train, test in kfold.split(data_X_true, data_Y_true):
        model = MimicModel(input_shape=(13,), action_shape=(2,))
        model.fit(data_X_true[train], data_Y_true[train], epoch=100, batch_size=256)
        scores = model.evaluate(data_X_true[test], data_Y_true[test], verbose=0)
        print("%s: %.2f%%" % (model.model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        model_candidates.append(model)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    best_index = np.argmax(cvscores)
    model = model_candidates[best_index]
    model.save("../saved/mimic_model.h5")


