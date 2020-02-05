import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers, regularizers
from tensorflow.python.keras.models import Sequential,load_model,Model
from tensorflow.python.keras.layers import Dense, Activation, Input, Multiply, Add
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
import pickle as pkl
import os
import pdb

def RL_func(input_dim, num_class):
    model = Sequential([
        Dense(500, input_shape=(input_dim,), kernel_initializer='he_normal', \
              kernel_regularizer=regularizers.l2(0.01), name="rl_model/d1"),
        Activation('relu', name="rl_model/r1"),
        Dense(500, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), name="rl_model/d2"),
        Activation('relu', name="rl_model/r2"),
        Dense(num_class, name="MnistModel/d3", kernel_regularizer=regularizers.l2(0.01))
    ])
    return model

# normalize the \ observation and action
def norm(obs):
    mean = np.mean(obs, axis=0)
    var = np.std(obs, axis=0)
    obs = (obs - mean) / (var + 1e-8)
    return mean, var, obs 

# load data
def load_data(filename):
    import pickle as pkl
    with open(filename, 'rb') as f:
        [opp_obs, opp_act, adv_act, opp_next_obs] = pkl.load(f)
    return opp_obs, opp_act, adv_act, opp_next_obs

class RL_model(object):
    def __init__(self, input_shape, out_shape, sess=None, graph=None):

        # input shape are same as output shape
        self.input_shape = input_shape
        self.num_class = out_shape[0]

        # todo add gpu support
        if sess == None:
          tf_config = tf.ConfigProto()
          self.session = tf.Session(config=tf_config)
          self.graph = tf.get_default_graph()
        else:
          self.session = sess
          self.graph=graph
        K.set_session(self.session)
        with self.graph.as_default():
            with self.session.as_default():
                self.model = self.build_model()

        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential([
            Dense(500, input_shape=(self.input_shape[0],), kernel_initializer='he_normal', \
                  kernel_regularizer=regularizers.l2(0.01), name="rl_model/d1"),
            Activation('relu', name="rl_model/r1"),
            Dense(500, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), name="rl_model/d2"),
            Activation('relu', name="rl_model/r2"),
            Dense(self.num_class, name="MnistModel/d3", kernel_regularizer=regularizers.l2(0.01))
        ])
        # todo define the optimization method
        adam = optimizers.Adam(lr=1e-5)
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])

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
    filename = '../agent-zoo/data/train_data'
    out_filename = './saved/opp_var'

    opp_obs, opp_act, adv_act, opp_next_obs = load_data(filename)
    
    data_X_true = np.concatenate((opp_obs, opp_act, adv_act), axis=1)
    data_Y_true = opp_next_obs
    
    x_mean, x_var, data_X_true = norm(data_X_true)
    y_mean, y_var, data_Y_true = norm(data_Y_true)
    with open(out_filename, 'wb') as writer:
      pkl.dump([x_mean, x_var, y_mean, y_var], writer, pkl.HIGHEST_PROTOCOL)
    writer.close()

    from sklearn.model_selection import KFold
    model_candidates = []

    kfold = KFold(n_splits=10, shuffle=True, random_state=1992)
    cvscores = []

    cnt = 0
    for train, test in kfold.split(data_X_true, data_Y_true):
        model = RL_model(input_shape=(395 + 17 + 17,), out_shape=(395,))
        model.fit(data_X_true[train], data_Y_true[train], epoch=50, batch_size=512)
        scores = model.evaluate(data_X_true[test], data_Y_true[test], verbose=0)
        print("%s: %.2f%%" % (model.model.metrics_names[1], scores[1] * 100))
        cvscores.append(1.0 * scores[1])
        model_candidates.append(model)
        out_file = './saved/mimic_model_' + str(cnt) + '.h5'
        model.save(out_file)
        cnt += 1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    best_index = np.argmin(cvscores)
    model = model_candidates[best_index]
    model.save("./saved/mimic_model.h5")
