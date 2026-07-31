#Neural Network
#Stoicastic gradient descent
#version 1.4.0

import numpy as np # noqa: I001
import random
import time
from mnist import MNIST

#main
    #Network call

#Network Class
    #dense_layer call
    #activation_layer call
    #cost_layer call

#Layer Classes
    #Foward
    #backprop
    #update

#to do:
#add backprop
#add update parameters

def main():
    size = (1,10,1)
    net = NETWORK(size)

    print(net.Layers)
    input_cache = np.array([[4], [3], [2], [-3]])
    output_cache = input_cache ** 2
    print(input_cache, "\n", output_cache)

    net.optimizer("STOICHASTIC_GRADIENT_DECSENT")
    net.telementary()
    net.train(input_cache, output_cache)

    net.telementary()
    net.test(input_cache, output_cache)


    mndata = MNIST('mnist')

    # training_images, training_labels = mndata.load_training()
    # testing_images, testing_labels = mndata.load_testing()

    # print(len(training_images), len(training_labels))
    # print(len(testing_images), len(testing_labels))

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), loss="MEAN_SQUARED_ERROR", activation="RELU"):
        self.loss = loss
        self.Layers = architecture(size, loss, activation)

    def train(self, input_cache, output_cache, epochs=2000, lr=0.001):
        if self.optimizer == "STOICHASTIC_GRADIENT_DECSENT":
            for epoch in range(epochs):
                for index in range(len(input_cache)):
                    inputs = input_cache[index]
                    outputs = output_cache[index]

                    #foward
                    self.foward_pass(inputs, outputs)

                    #backward
                    self.backprop(outputs)

                    #update parameters
                    self.update_parameters(lr)

        elif self.optimizer == "GRADIENT_DECSENT":
            for epoch in range(epochs):
                #set gradients to zero
                for index in range(len(input_cache)):
                    inputs = input_cache[index]
                    outputs = output_cache[index]

                    #foward
                    x = inputs
                    for layer in self.Layers:
                        x = layer.foward(x)

                    #backward
                    following_layer = self.Layers[-1]
                    grad = following_layer.backprop(outputs)
                    
                    for index, layer in enumerate(reversed(self.Layers)):
                        if not index == 0:
                            grad = layer.backprop(following_layer)
                            following_layer = layer

                    #update gradients

                #update parameters
        
        elif self.optimizer == "MOMENTUM":
            #momentum gradient descent here
            pass

        else:
            print("Please enter a optimizer for training")

    def foward_pass(self, inputs, outputs):
        x = inputs
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)
        self.Layers[-1].foward(x, outputs)

    def foward_return(self, inputs, outputs):
        x = inputs
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)
        predicted = x
        
        cost = self.Layers[-1].foward(predicted, outputs)
        return cost, predicted

    def telementary(self):
        for layer in self.Layers:
            if layer.type == "Dense":
                layer.telementary_parameters()

    def backprop(self, outputs):
        following_layer = self.Layers[-1]
        following_layer.backprop(outputs)
        
        for index, layer in enumerate(reversed(self.Layers)):
            if not index == 0:
                layer.backprop(following_layer)
                following_layer = layer

    def update_parameters(self, lr):
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                layer.update_parameters(lr)

    def test(self, input_cache, output_cache):
        for index, inputs in enumerate(input_cache):
            outputs = output_cache[index]
            cost, predicted = self.foward_return(inputs, outputs)

            try:
                predicted[0][0]
            except:
                print(f"Predicted: {predicted[0]:.4f}    Actual: {outputs[0]:.6f}    Cost: {cost:.6f}")
            else:
                print(f"Predicted: {predicted[0][0]:.4f}    Actual: {outputs[0]:.6f}    Cost: {cost:.6f}")

    def optimizer(self, type):
        if type == "STOICHASTIC_GRADIENT_DECSENT":
            self.optimizer = type
        elif type == "GRADIENT_DECSENT":
            self.optimizer = type
        elif type == "MOMENTUM":
            self.optimizer = type

class DENSE_LAYER:
    def __init__(self, num_inputs, num_outputs, activation_type):
        self.weights = np.random.randn(num_outputs, num_inputs)
        self.biases = np.random.randn(num_outputs, 1)
        self.activation = activation_type
        self.type = "Dense"

    def foward(self, inputs):
        self.inputs = inputs
        z = np.dot(self.weights, inputs)
        z += self.biases
        a = None
        if self.activation == "RELU":
            a = np.max(z, 0)
        elif self.activation == "SIGMOID":
            a =  1 / (1 + np.exp(-z))
        elif self.activation == "NONE":
            a = z
        self.weighted_sums = z
        self.outputs = a
        return a

    def telementary_parameters(self):
        print("weights: ", end="")
        print(self.weights)
        print("biases: ", end="")
        print(self.biases)

    def backprop(self, following_layer):
        partial_derivative = None
        if self.activation == "RELU":
            partial_derivative = (self.weighted_sums > 0).astype(dtype=float)
        elif self.activation == "SIGMOID":
            partial_derivative = self.outputs * ( 1 - self.outputs)
        elif self.activation == "NONE":
            partial_derivative = 1

        aft_deltas = following_layer.deltas
        if not following_layer.type == "Cost":
            aft_weights = following_layer.weights

            self.deltas = np.transpose(aft_weights) @ aft_deltas * partial_derivative
        else:
            self.deltas = aft_deltas * partial_derivative
        
    def update_parameters(self, lr):
        self.biases = self.biases - lr * self.deltas

        self.weights = self.weights - lr * np.outer(self.inputs, self.deltas)

class COST_LAYER:
    def __init__(self, cost_type):
        self.cost = cost_type
        self.type = "Cost"

    def foward(self, inputs, outputs):
        self.predicted = inputs
        c = None
        if self.cost == "MEAN_SQUARED_ERROR":
            c = np.sum((outputs - inputs) ** 2)
        elif self.cost == "CROSS_ENTROPY":
            c = cross_entropy(outputs, inputs)
        self.outputs = c
        return c

    def backprop(self, outputs):
        if self.cost == "MEAN_SQUARED_ERROR":
            self.deltas = 2 * (self.predicted - outputs)
        elif self.cost == "CROSS_ENTROPY":
            pass
  
def architecture(size, loss, activation_type):
    Layers = []
    for index in range(len(size) - 1):
        Layers.append(DENSE_LAYER(size[index], size[index + 1], activation_type))
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
