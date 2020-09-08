import numpy as np
from tensorflow.python.keras import backend as K
import pdb

# this function serve as a benchmark

class GradientExp(object):
    def __init__(self, model, sess=None):
        K.set_learning_phase(0)
        self.model = model

        with self.model.graph.as_default():
            with self.model.session.as_default():
                self.class_grads = K.function([self.model.model.input],
                                              K.gradients(self.model.model.output, self.model.model.input))
                self.out = K.function([self.model.model.input], self.model.model.output)

    def output(self, x):
        with self.model.graph.as_default():
            with self.model.session.as_default():
                out_v = self.out([x])

        return out_v

    def grad(self, x, normalize=True):
        with self.model.graph.as_default():
            with self.model.session.as_default():
                sal_x = self.class_grads([x])[0]
                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x

    def integratedgrad(self, x, x_baseline=None, x_steps=25, normalize=True):
        with self.model.graph.as_default():
            with self.model.session.as_default():

                if x_baseline is None:
                    x_baseline = np.zeros_like(x)
                else:
                    assert x_baseline.shape == x.shape

                x_diff = x - x_baseline
                total_gradients = np.zeros_like(x)

                for alpha in np.linspace(0, 1, x_steps):
                    x_step = x_baseline + alpha * x_diff
                    grads = self.class_grads([x_step])[0]
                    total_gradients += grads
                sal_x = total_gradients * x_diff

                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x

    def smoothgrad(self, x, stdev_spread=0.1, nsamples=25, magnitude=True, normalize=True):

        with self.model.graph.as_default():
            with self.model.session.as_default():

                stdev = stdev_spread * (np.max(x) - np.min(x))
                total_gradients = np.zeros_like(x)

                for i in range(nsamples):
                    noise = np.random.normal(0, stdev, x.shape)
                    x_plus_noise = x + noise
                    grads = self.class_grads([x_plus_noise])[0]

                    if magnitude:
                        total_gradients += (grads * grads)
                    else:
                        total_gradients += grads

                sal_x = total_gradients / nsamples

                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x
