#Neural Network with GPU and possibly TPU
#version 1.2.1

import numpy as cp # noqa: I001
from mnist import MNIST

def main():
    input_array = cp.array(range(10))
    output_array = 2 * input_array + 1
    
    NNM = network(1, 1, 1, 1)
    NNM.feed(input_array, output_array)
    NNM.train(1000, 0.05)

    draw_mnist_digit(7)




class network():
    def __init__(self, a, b, c, d):
        self.input_amt = a
        self.hidden_layer_amt = b
        self.neuron_amt = c
        self.output_amt = d

    def train(self, iterations, lr):
        for epoch in range(iterations):
            #pass
            #back prop
            #update parameters
            pass

    def feed(self, array_1, array_2):
        self.input_cache = array_1
        self.actual_cache = array_2

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