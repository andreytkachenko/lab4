import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist
import math as m
from scipy import signal

np.random.seed(1)                   # заставим numpy выдавать одинаковые набор случайных чисел для каждого запуска программы
np.set_printoptions(suppress=True)  # выводить числа в формате 0.123 а не 1.23e-1

# В `X` находятся изображения для обучения, а в `y` значения соответственно
# `X.shape` == (60000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `y.shape` == (60000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
X, y = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/train-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/train-labels-idx1-ubyte")

# В `Xt` находятся изображения для тестирования, а в `yt` значения соответственно
# `Xt.shape` == (10000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `yt.shape` == (10000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
Xt, yt = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/t10k-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/t10k-labels-idx1-ubyte")

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1.0 / (1.0 + np.exp(-x))

def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d

X = X * (1 / 255)
Xt = Xt * (1 / 255)

# Параметры:

lr = 0.01     # значени на которое будет домножаться дельта на каждом шаге
batch = 60    # кол-во изображений использованное для обучения на каждом шаге
epochs = 100  # кол-во эпох. Если видно что прогресс есть, но нужно больше итераций 


def convolution(X, W):
    (filter_channels, filter_height, filter_width) = W.shape
    (batch_size, in_channels, in_rows, in_cols) = X.shape
    (out_channels, out_rows, out_cols) = (filter_channels, in_rows - 2, in_cols - 2)

    res = np.zeros((batch_size, out_channels, out_rows, out_cols))
 
    for och in range(0, out_channels):
        for batch in range(0, batch_size):
            for ich in range(0, in_channels):
                res[batch][och] += signal.convolve2d(X[batch][ich], W[och], mode='valid')

    return res


class MnistConvModel:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.filters = 8

        self.W_conv = np.random.uniform(-0.05, 0.05, (self.filters, 3, 3))
        self.W_linear = np.random.uniform(-0.05, 0.05, (self.filters * 26 * 26, 10))
        self.b_linear = np.zeros((10, ))

    def load(self, conv, linear):
        with open(conv, 'rb') as f:
            self.W_conv = np.array(pickle.load(f)).reshape((self.filters, 3, 3))

        with open(linear, 'rb') as f:
            self.W_linear = np.array(pickle.load(f)).reshape((self.filters * 26 * 26, -1))

    # Linear Layer

    def linear_forward(self, X):
        return np.dot(X, self.W_linear) + self.b_linear

    def linear_backward(self, e):
        return np.dot(e, self.W_linear.T)

    # Sigmoid Layer

    def sigmoid_forward(self, X):
        return sigmoid(X)

    def sigmoid_backward(self, e):
        return e * sigmoid(self.o_sigmoid, True)

    # ReLU Layer

    def relu_forward(self, X):
        X_o = X.copy()
        X_o[X < 0] = 0

        return X_o
    
    def relu_backward(self, e):
        res = self.o_relu.copy()
        res[res > 0] = 1.0

        return e * res

    # Convolution Layer

    def convolution_forward(self, X):
        return convolution(X, self.W_conv)


    def convolution_backward(self, e):
        return e


    def forward(self, X):
        self.X = X
        
        self.o_conv = self.convolution_forward(X)
        
        self.o_relu = self.relu_forward(self.o_conv)

        self.o_relu_flatten = self.o_relu.reshape(len(X), -1)
        
        self.o_linear = self.linear_forward(self.o_relu_flatten)
        
        self.o_sigmoid = self.sigmoid_forward(self.o_linear)
        
        return self.o_sigmoid


    def backward(self, e):
        self.e_sigmoid = self.sigmoid_backward(e)

        self.e_linear = self.linear_backward(self.e_sigmoid)

        self.e_relu = self.relu_backward(self.e_linear.reshape((-1, self.filters, 26, 26)))
        
        # self.e_conv = self.convolution_backward(self.e_relu)

    def calc_gradients(self):
        def conv(X, e):
            (out_batch_size, out_channels, out_rows, out_cols) = e.shape
            (batch_size, in_channels, in_rows, in_cols) = X.shape
            (filters, w_rows, w_cols) = (out_channels, 3, 3)

            res = np.zeros((filters, w_rows, w_cols))
        
            for och in range(0, out_channels):
                for batch in range(0, batch_size):
                    for ich in range(0, in_channels):
                        res[och] += signal.convolve2d(X[batch][ich], e[batch][och], mode='valid')

            return res

        scaler = 1 / len(self.X)

        self.dW_linear = np.dot(self.o_relu_flatten.T, self.e_sigmoid) * scaler
        self.db_linear = np.mean(self.o_linear, axis=0) * scaler
        self.dW_conv = conv(self.X, self.e_relu) * scaler

    def update(self):
        self.W_linear -= self.dW_linear * self.lr
        self.W_conv -= self.dW_conv * self.lr


def mse(o, y):
    return np.sum(np.square(o - y))

def mse_prime(o, y):
    return 2 * (o - y)

def validate(model, X, y):
    tp = model.forward(X)

    return np.sum(y == np.argmax(tp, axis=1)) / len(y)

def train(model, X, y, epochs=100, batch_size=100, validation=None):
    batch_count = m.ceil(len(y) / batch_size)
    
    t = np.zeros((len(y), 10))
    np.put_along_axis(t, y.reshape((-1, 1)), 1.0, axis=1)

    for epoch in range(0, epochs):
        print("Epoch ", epoch + 1)

        for index, (bX, bt) in enumerate(zip(np.split(X, batch_count), np.split(t, batch_count))):
            res = model.forward(bX)
            error = mse_prime(res, bt)

            model.backward(error)
            model.calc_gradients()
            model.update()

            if index % 100 == 0:
                print("  Loss: ", mse(res, bt))

        if validation is not None:
            (val_X, val_y) = validation
            print("  Accuracy: ", validate(model, val_X, val_y))

if __name__ == "__main__":
    model = MnistConvModel(lr=0.1)
    X = X.reshape((-1, 1, 28, 28))
    Xt = Xt.reshape((-1, 1, 28, 28))

    train(model, X, y, validation=(Xt, yt))

