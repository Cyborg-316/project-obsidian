#Neural Network with GPU and possibly TPU
#Stoicastic gradient descent
#version 1.3.2

import numpy as cp # noqa: I001
import random
from mnist import MNIST

def main():
    input_array = cp.array(range(10))
    output_array = 3 * input_array + 10
    print(input_array, output_array)
    
    Layers = [LAYER(1, 1), LAYER(1, 1)]

    print("weight(s):", Layers[0].weights, "bias(es):", Layers[0].biases)


    for epoch in range(400):
        for index in range(len(input_array)):
            # print("running")
            #foward pass
            x = cp.array([[input_array[index]]])
            for layer in Layers:
                x = layer.foward(x)

            actual = cp.array([[output_array[index]]])

            #back prop
            Layers[-1].backward1(actual)
            # for layer in reversed(Layers):
            #     gradient = layer.backward2(gradient)
            
            #update
            for layer in Layers:
                layer.update_parameters()

    print("weight(s):", Layers[0].weights, "bias(es):", Layers[0].biases)
    Cost = 0
    for index in range(len(input_array)):
        # print("running")
        #foward pass
        x = cp.array([[input_array[index]]])
        for layer in Layers:
            x = layer.foward(x)


        actual = cp.array([[output_array[index]]])
        print("pred", x, "actual", actual)
        Cost = (x - actual) ** 2

    print(Cost)



    # draw_mnist_digit(random.randint(0,9))

class LAYER():
    def __init__(self, a, b):
        self.input_amt = a
        self.output_amt = b

        #init weights & biases
        if b > 1 or a > 1:
            self.weights = CUSTOM_ARRAY.random(b, a)
            self.biases = CUSTOM_ARRAY.zeros(b, 1)
        else:
            self.weights = CUSTOM_ARRAY.single1()
            self.biases = CUSTOM_ARRAY.single2()

    def foward(self, inputs):
        #return activations and store weighted sums, inputs, and dA of z
        self.prev_activations = inputs
        z = cp.dot(self.weights, inputs)
        if self.output_amt == 1:
            z = cp.array([z])
        z += self.biases
        a = self.activation(z)
        self.weighetd_sums = z
        self.activations = a
        self.d_activations = (a > 0).astype(float)
        return a

    def backward1(self, outputs):
        grad = 2 * (self.activations - outputs) #* self.d_activations
        self.gradient = grad
        return grad

    def backword2(self, gradient, after_weights):
        grad = cp.dot()

    def update_parameters(self):
        lr = 0.01
        grad = self.gradient

        if self.input_amt > 1 or self.output_amt > 1:
            self.weights -= lr * (grad @ cp.transpose(self.prev_activations))
            self.biases -= lr * grad
        else:
            self.weights -= lr * (grad * self.prev_activations)[0]
            self.biases -= lr * grad[0]


    def activation(self, x):
        return cp.maximum(x, 0)

class CUSTOM_ARRAY():
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
