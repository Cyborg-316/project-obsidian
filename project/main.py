#Neural Network with GPU and possibly TPU
#Stoicastic gradient descent
#version 1.3.1

import numpy as cp # noqa: I001
import random
from mnist import MNIST

def main():
    input_array = cp.array(range(10))
    output_array = 2 * input_array + 1
    print(input_array, output_array)
    
    Layers = [LAYER(1, 1)]

    print("weight(s):", Layers[0].weights, "bias(es):", Layers[0].biases)

    x = cp.array([[input_array[1]]])
    for layer in Layers:
        x = layer.foward(x)
    print("a", x)





    draw_mnist_digit(random.randint(0,9))

class LAYER():
    def __init__(self, a, b):
        self.input_amt = a
        self.output_amt = b

        #init weights & biases
        if b > 1 or a > 1:
            self.weights = custom_array.random(b, a)
            self.biases = custom_array.zeros(b, 1)
        else:
            self.weights = custom_array.single1()
            self.biases = custom_array.single2()

    def foward(self, inputs):
        #return activations, store weighted sums
        z = cp.dot(self.weights, inputs)
        if self.output_amt == 1:
            z = cp.array([z])
        z += self.biases
        a = self.activation(z)
        self.weighetd_sums = z
        self.activations = a
        return a

    def activation(self, x):
        return cp.maximum(x, 0)

    def d_activation(self, x):
        return (x > 0).astype(float)

class custom_array():
    def random(rows, columns):
        return cp.random.randn(rows, columns) * cp.sqrt(2 / columns)

    def zeros(rows, columns):
        return (cp.random.randint(3, size=(rows, columns)) - 1) * cp.sqrt(2 / columns)

    def ones(rows, columns):
        return cp.ones((rows, columns))

    def single1():
        return cp.array([cp.random.randn() * cp.sqrt(2)])

    def single2():
        return cp.array([cp.sqrt(2)]) * (random.randint(-1,1))

def draw_mnist_digit(number):
    mndata = MNIST('mnist')
    # train_images = images[:50000]
    # train_labels = labels[:50000]
    # test_images = images[50000:]
    # test_labels = labels[50000:]

    images, labels = mndata.load_training()

    index = labels.index(number)
    data = images[index]

    for x in range(28):
        print("|", end = "")
        for y in range(28):
            draw_char(data[y + 28 * x])
        print("|")

def draw_char(char):
    if char > 0:
        print("0 ", end = "")
    else:
        print("  ", end = "")

main()
