import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import tensorflow as tf
import explain_hua_gradient as exp
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

from RoboschoolPong_v0_2017may2 import SmallReactivePolicy

class MasterModel(object):
    weights_dense1_w_const = tf.constant(SmallReactivePolicy.weights_dense1_w, dtype=tf.float32)
    weights_dense1_b_const = tf.constant(SmallReactivePolicy.weights_dense1_b, dtype=tf.float32)
    weights_dense2_w_const = tf.constant(SmallReactivePolicy.weights_dense2_w, dtype=tf.float32)
    weights_dense2_b_const = tf.constant(SmallReactivePolicy.weights_dense2_b, dtype=tf.float32)
    weights_final_w_const = tf.constant(SmallReactivePolicy.weights_final_w, dtype=tf.float32)
    weights_final_b_const = tf.constant(SmallReactivePolicy.weights_final_b, dtype=tf.float32)

    # x = tf.nn.relu(tf.matmul(x, weights_dense1_w_const) + weights_dense1_b_const)
    # x = tf.nn.relu(tf.matmul(x, weights_dense2_w_const) + weights_dense2_b_const)
    # output_action = tf.matmul(x, weights_final_w_const) + weights_final_b_const

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

    def build_model(self):

        with self.graph.as_default():
            with self.session.as_default():
                model = Sequential([
                    Dense(64, input_shape=(13,),name="MnistModel/d1", trainable=False,
                          weights=[SmallReactivePolicy.weights_dense1_w, SmallReactivePolicy.weights_dense1_b]),
                    Activation('relu', name="MnistModel/r1",trainable=False),
                    Dense(32, name="MnistModel/d2", trainable=False,
                          weights=[SmallReactivePolicy.weights_dense2_w, SmallReactivePolicy.weights_dense2_b]),
                    Activation('relu', name="MnistModel/r2", trainable=False),
                    Dense(2, name="MnistModel/d3", trainable=False,
                          weights=[SmallReactivePolicy.weights_final_w, SmallReactivePolicy.weights_final_b])
                ])

                model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

        return model

    def fit(self, x_train, y_train, batch_size, epoch):

        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit(x = x_train,
                               y = y_train,
                               batch_size = batch_size,
                               epochs = epoch)
        return 0

    def evaluate(self, x, y, batch_size=32, verbose = 0):
        with self.graph.as_default():
            with self.session.as_default():
                if x.ndim ==3:
                    x = np.expand_dims(x, 0)
                loss, acc = self.model.evaluate(x = x, y = y, batch_size= batch_size, verbose = verbose)
        return loss, acc

    def predict(self, x, batch_size=32, verbose = 0):
        with self.graph.as_default():
            with self.session.as_default():
                if x.ndim ==3:
                    x = np.expand_dims(x, 0)
                pred = self.model.predict(x = x, batch_size = batch_size, verbose = verbose)
        return pred

    def save(self, model_url):
        self.model.save(model_url)
        return 0

    def load(self, model_url):
        self.model = load_model(model_url)
        return 0

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MnistModel')

    def set_variables(self):
        self.model.layers[0].set_weights([SmallReactivePolicy.weights_dense1_w, SmallReactivePolicy.weights_dense1_b])
        self.model.layers[2].set_weights([SmallReactivePolicy.weights_dense2_w, SmallReactivePolicy.weights_dense2_b])
        self.model.layers[4].set_weights([SmallReactivePolicy.weights_final_w, SmallReactivePolicy.weights_final_b])


if __name__ == "__main__":

    # model and data hypers
    BATCH_SIZE = 32
    EPOCH = 10

    # explanation hypers
    LAMBDA_1 = 1e-1
    LAMBDA_2 = 1e-2
    OPTIMIZER = tf.train.AdamOptimizer
    INITIALIZER = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    LR = 1e-2
    REGULARIZER = 'l1'
    MASK_SHAPE = (13,)
    EXP_BATCH_SIZE = 32
    EXP_EPOCH = 500
    EXP_DISPLAY_INTERVAL = 20
    EXP_LAMBDA_PATIENCE = 20
    EARLY_STOP_PATIENCE = 50

    def load_data(filename):
        import pickle as pkl

        def pickleLoader(pklFile):
            try:
                while True:
                    yield pkl.load(pklFile)
            except EOFError:
                pass

        total_data_X = []
        total_data_Y = []
        with open(filename, "rb") as f:
            for event in pickleLoader(f):
                total_data_X.append(event[0])
                total_data_Y.append(event[1])
        return np.array(total_data_X)[0,:], np.array(total_data_Y)[0,:]


    data_X, data_Y = load_data("/Users/huawei/PycharmProjects/AdvMARL/log/for_exp/test-wenbo.pkl")

    with tf.Session() as sess:
        model = MasterModel(input_shape=(13,), action_shape=(2,))
        exp_test = exp.GradientExp(model.model)
        mask_1 = exp_test.grad(data_X, data_Y)
        mask_2 = exp_test.integratedgrad(data_X, data_Y)
        mask_3 = exp_test.smoothgrad(data_X, data_Y)

        mask_best_all_acc = np.hstack((mask_1, mask_2, mask_3, data_X))
        np.savetxt("/Users/huawei/PycharmProjects/AdvMARL/log/for_exp/explained-X-gradient-new.csv", mask_best_all_acc,
                   delimiter=",")