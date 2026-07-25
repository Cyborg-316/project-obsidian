#Neural Network
#Stoicastic gradient descent
#version 1.4.0

import numpy as np # noqa: I001
import random
import time
from mnist import MNIST


def main():
    size = (1,10,10, 5, 5)
    types = ["Dense", "Activation", "Dense", "Activation", "Cost"]
    net = NETWORK()

    print(net.Layers)
    inputs = np.arange(5)
    outputs = inputs * 4 + 1



    mndata = MNIST('mnist')

    # training_images, training_labels = mndata.load_training()
    # testing_images, testing_labels = mndata.load_testing()

    # print(len(training_images), len(training_labels))
    # print(len(testing_images), len(testing_labels))

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), types=["Dense"], loss="MEAN_SQUARED_ERROR", activation="RELU"):
        self.loss = loss
        self.Layers = architecture(size, types, loss, activation)

    def train(self, input_cache, output_cache, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            for index in range(len(input_cache)):
                inputs = input_cache[index]
                outputs = output_cache[index]

                x = inputs
                for layer in self.Layers:
                    x = layer.foward(x)

                grad = Layers[-1].backprop()

            #update

    def test(self, input_cache, output_cache):
        pass

    def optimizer(self):
        pass

class DENSE_LAYER:
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.randn(num_outputs, num_inputs)
        self.biases = np.random.randn(num_outputs, 1)

    def foward(self, inputs):
        z = self.weights @ inputs + self.biases
        self.inputs = inputs
        self.outputs = z
        return z

    def backprop(self, following_layer):
        pass

class ACTIVATION_LAYER:
    def __init__(self, activation_type):
        self.activation = activation_type

    def foward(self, inputs, type):
        self.inputs = inputs
        a = None
        if self.activation == "RELU":
            a = np.max(inputs, 0)
        elif self.activation == "SIGMOID":
            a =  1 / (1 + np.exp(-inputs))
        self.outputs = a
        return a

    def backprop(self, following_layer):
        pass

class COST_LAYER:
    def __init__(self, cost_type):
        self.cost = cost_type

    def foward(self, inputs, outputs):
        self.inputs = inputs
        c = None
        if self.cost == "MEAN_SQUARED_ERROR":
            c = np.sum(np.sub(outputs, inputs) ** 2)
        elif self.cost == "CROSS_ENTROPY":
            c = cross_entropy(softmax(inputs), outputs)
        self.outputs = c
        return c

    def backprop(self, outputs):
        pass
   
def architecture(size, types, loss, activation_type):
    Layers = []
    for index, type in enumerate(types):
        if type == "Dense":
            Layers.append(DENSE_LAYER(size[index], size[index + 1]))
        elif type == "Activation":
            Layers.append(ACTIVATION_LAYER(activation_type))
        elif type == "Cost":
            Layers.append(COST_LAYER(loss))
    return Layers

def softmax(matrix):
    mat = np.exp(matrix)
    sum = np.sum(mat)
    return mat * (1 / sum)

def cross_entropy(matrix_p, matrix_q):
    if (matrix_p.shape != matrix_q.shape):
        return False

    mat = np.multiply(matrix_p, np.log(matrix_q))
    return -np.sum(mat)

def draw_mnist_digit(image_list):
    for x in range(28):
        print("|", end = "")
        for y in range(28):
            char = image_list[y + 28 * x]
            if char > 0:
                    print("0 ", end = "")
            else:
                print("  ", end = "")
        print("|")

start = time.perf_counter()
print("\n---------------RUNNING---------------\n\n\n\n\n")
main()
end = time.perf_counter()
print(f"\n-----FINISHED--IN--{(end - start):.9f}s------\n")
