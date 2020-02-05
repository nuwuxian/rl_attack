import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
import pdb

# define the rl function
def RL_func(input_dim, out_class):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), kernel_initializer='random_uniform', bias_initializer='zeros',
              name="rl_model/d1"),
        Activation('relu', name="rl_model/r1"),
        Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', name="MnistModel/d2"),
        Activation('relu', name="rl_model/r2"),
        Dense(out_class, kernel_initializer='random_uniform', bias_initializer='zeros', name="rl_model/d3")
    ])
    return model

def Linf_error(y_true, y_pred):
    return K.mean(K.square(K.maximum(K.abs(y_true - y_pred) - 0.05, 0)), axis=-1)

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

        # initilize the model
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential([
            Dense(64, input_shape=(self.input_shape[0],), kernel_initializer='random_uniform', bias_initializer='zeros', name="MnistModel/d1"),
            Activation('relu', name="MnistModel/r1"),
            Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', name="MnistModel/d2"),
            Activation('relu', name="MnistModel/r2"),
            Dense(self.num_class[0], kernel_initializer='random_uniform', bias_initializer='zeros', name="MnistModel/d3")
        ])
        model.compile(optimizer='rmsprop', loss=Linf_error, metrics=[Linf_error])

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
        self.model = load_model(model_url, custom_objects={'Linf_error':Linf_error})
        return 0