#Neural Network with GPU and possibly TPU
#Stoicastic gradient descent
#version 1.3.6

import numpy as cp # noqa: I001
import random
from mnist import MNIST

def main():
    print("---------------RUNNING---------------\n\n\n\n\n")


    temp = 10
    TRAIN_NETWORK(
        inputs=cp.array(range(temp)), 
        outputs=cp.array(range(temp))**2, 
        size=(1,temp,1), 
        COST="MEAN_SQUARED_ERROR",
        learning_rate=0.001,
        iterations=100000,
        telementary=False
    )
    



    # draw_mnist_digit(random.randint(0,9))
    print("\n\n\n\n\n---------------FINISHED--------------")

def TRAIN_NETWORK(inputs, outputs, size, COST, learning_rate, iterations, telementary):
    input_array = inputs
    output_array = outputs

    NET = NETWORK(ARCHITECTURE(size), COST)

    if telementary:
        NET.TELEMENTARY_PARAMETERS()
        print("---------------TRAINING--------------")

    for epoch in range(iterations):
        for index in range(len(input_array)):
            NET.FOWARD(cp.array([[input_array[index]]]))

            NET.BACKPROP(actual=cp.array([[output_array[index]]]), COST="MEAN_SQUARED_ERROR")

            NET.UDPATE(learning_rate)
    
    if telementary:
        print("---------------COMPLETED-------------")
        NET.TELEMENTARY_PARAMETERS()
    NET.TELEMENTARY_PRED_VS_ACTUAL(inputs=input_array, outputs=output_array)

def ARCHITECTURE(Layout):
    neurons = list(Layout)
    Layers = []
    for i in range(len(neurons) - 1):
        Layers.append(LAYER(neurons[i], neurons[i+1], i, len(neurons) - 1))
    return Layers

class NETWORK():
    def __init__(self, layers, COST):
        self.Layers = layers
        self.cost = COST

    def FOWARD(self, inputs):
        x = inputs
        for layer in self.Layers:
            x = layer.foward(x)

    def BACKPROP(self, actual, COST):
        Layers = self.Layers
        gradient = None
        for layer in reversed(Layers):
            gradient = layer.backward(gradient, Layers, actual, COST)

    def UDPATE(self, learning_rate):
        for layer in self.Layers:
            layer.update_parameters(learning_rate)

    def TELEMENTARY_PARAMETERS(self):
        Layers = self.Layers
        for layer in Layers:
            print("weight(s):", layer.weights, "bias(es):", layer.biases, "Layer index:", Layers.index(layer))

    def TELEMENTARY_PRED_VS_ACTUAL(self, inputs, outputs):
        Cost = 0
        for index in range(len(inputs)):
            # print("running")
            #foward pass
            self.FOWARD(cp.array([[inputs[index]]]))
            pred = self.Layers[-1].activations
            actual = cp.array([[outputs[index]]])
            print("pred", pred, "actual", actual)
            Cost += (pred - actual) ** 2
        print(f"{self.cost}: {Cost[0][0]:.3f}")

class LAYER():
    def __init__(self, a, b, index, layer_amt):
        self.input_amt = a
        self.output_amt = b
        self.layer_index = index
        self.total_layer_amt = layer_amt

        #init weights & biases
        if b > 1 or a > 1:
            self.weights = CUSTOM_ARRAY_OPERATIONS.create_random(b, a)
            self.biases = CUSTOM_ARRAY_OPERATIONS.create_random(b, 1)
        else:
            self.weights = CUSTOM_ARRAY_OPERATIONS.create_single1()
            self.biases = CUSTOM_ARRAY_OPERATIONS.create_single2()

    def foward(self, inputs):
        self.prev_activations = inputs
        
        z = cp.dot(self.weights, inputs) + self.biases
        a = self.activation(z)

        self.weighetd_sums = z
        self.activations = a
        self.d_activations = self.d_activation(z)
        self.total_grad = CUSTOM_ARRAY_OPERATIONS.create_zeros(self.output_amt, 1)
        return a

    def backward(self, gradient, Layers, outputs, COST):
        if self.layer_index == self.total_layer_amt - 1:
            if COST == "MEAN_SQUARED_ERROR":
                    grad = 2 * (self.activations - outputs) #* self.d_activations
                    self.gradient = grad
                    return grad
            elif COST == "CROSS_ENTROPY_LOSS":
                pass

        after_weights = Layers[Layers.index(self) + 1].weights
        grad = cp.dot(after_weights.T, gradient).reshape(self.output_amt, self.input_amt) * self.d_activations
        self.total_grad += grad
        return grad

    def update_parameters(self, lr):
        grad = self.total_grad

        if self.input_amt > 1 or self.output_amt > 1:
            self.weights -= lr * (grad @ cp.transpose(self.prev_activations))
            self.biases -= lr * grad
        else:
            self.weights -= lr * (grad * self.prev_activations)[0]
            self.biases -= lr * grad[0]

    def activation(self, x):
        return cp.maximum(x, 0)

    def d_activation(self, x):
        return (x > 0).astype(float)

class CUSTOM_ARRAY_OPERATIONS():
    def create_random(rows, columns):
        return cp.random.randn(rows, columns) * cp.sqrt(2 / rows)

    def create_zeros(rows, columns):
        return cp.zeros((rows, columns))

    def create_ones(rows, columns):
        return cp.ones((rows, columns))

    def create_single1():
        return cp.array([cp.random.randn() * cp.sqrt(2)])

    def create_single2():
        return cp.array([cp.sqrt(2)]) * (random.randint(-1,1))

    def softmax(matrix):
        mat = cp.exp(matrix)
        sum = mat_sum(mat)
        return mat * (1 / sum)

    def cross_entropy(matrix_p, matrix_q):
        if (matrix_p.shape != matrix_q.shape):
            return False

        mat = cp.multiply(matrix_p, cp.log(matrix_q))
        return -cp.sum(mat)

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
