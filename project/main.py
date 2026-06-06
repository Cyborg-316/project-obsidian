#Neural Network with GPU and possibly TPU
#version 1.0.9

import numpy as cp
#import math
#import pygame
#import random
#import cupy as cp
#from mnist import MNIST

def main():
    __onStart()

    # mndata = MNIST('mnist')

    # images, labels = mndata.load_training()
    # train_images = images[:50000]
    # train_labels = labels[:50000]
    # test_images = images[50000:]
    # test_labels = labels[50000:]

    # draw_mnist_digit(train_images[900])
    # print(train_labels[900])
    #print(num_to_vector(train_labels[7]))

    #Layer1 = structure(784, 128)
    #Layer2 = structure(128, 10)
    #128 neurons in 1 hideen layer with 784 inputs and 10 outputs








    global learning_rate
    learning_rate = 0.005

    inputs = cp.arange(0,10, 1, dtype = float)
    outputs = cp.arange(0,20, 2, dtype = float) ** 2


    Layer1 = structure(1, 10, "hidden_layer")
    Layer2 = structure(10, 1, "last_layer")

    for epoch in range(1):
        Layer1.reset_gradients()
        Layer2.reset_gradients()

        for i in range(len(inputs)):
            Layer1.input(cp.array([inputs[i]]))
            Layer2.input(Layer1.activations())
            Layer2.expected(mat_transpose(cp.array([outputs[i]])))
            Layer1.after_weights(Layer2.weights)
            Layer1.after_deltas(Layer2.deltas())

            print("input:", inputs[i],"actual:", outputs[i], "     pred:",Layer2.activations(),  Layer2.deltas())

            Layer1.update_gradients()
            Layer2.update_gradients()
        
        print(Layer1.bias_gradients)
        Layer1.update_parameters()
        Layer2.update_parameters()


    print(Layer1.weights, "x + ", Layer1.biases)
    print(Layer2.weights, "x + ", Layer2.biases)

    #REMEBER TO FEED IT THE SOFT MAX VERSIONS FOR CROSS ENTROPY
    __onEnd()

class structure:
    def __init__(self, num_input_neurons, num_output_neurons, layer_type):
        self.weights = cp.ones((num_output_neurons, num_input_neurons))
        self.biases = mat_transpose(cp.ones(num_output_neurons))
        self.layer_type = layer_type
        self.weight_gradients = cp.zeros((num_output_neurons, num_input_neurons))
        self.bias_gradients = mat_transpose(cp.zeros(num_output_neurons))

    def input(self, inputs):
        self.inputs = inputs

    def expected(self, outputs):
        self.actual = outputs

    def after_weights(self, succeeding_weights):
        self.succeeding_weights = succeeding_weights

    def after_deltas(self, succeeding_deltas):
        self.succeeding_deltas = succeeding_deltas

    def weighted_sums(self):
        z = mat_transpose(self.weights @ self.inputs) + self.biases
        return z

    def activations(self):
        return mat_relu(self.weighted_sums())

    def soft_max(self):
        mat = self.activations()
        mat = mat_softmax(mat)
        return mat

    def cross_entropy(self):
        mat = self.soft_max()
        mat = mat_cross_entropy(mat, self.actual)
        return mat

    def deltas(self):
        if self.layer_type == "last_layer":
            mat = 2 * mat_sub(self.activations(), self.actual)
            return mat
        elif self.layer_type == "hidden_layer":
            mat_a = self.succeeding_weights
            mat_b = mat_transpose(self.succeeding_deltas)
            mat_c = mat_d_relu(self.weighted_sums())

            return mat_mult(mat_c, mat_dot(mat_a, mat_b, True, False), False, False)


    def update_parameters(self):
        self.weights = mat_add(self.weights, -learning_rate * self.weight_gradients)
        self.biases = mat_add(self.biases, -learning_rate * self.bias_gradients)

    def update_gradients(self):
        #WEIGHT UPDATES ARE WRONG
        self.weight_gradients = mat_add(self.bias_gradients, self.deltas())
        self.bias_gradients = mat_add(self.bias_gradients, self.deltas())

    def reset_gradients(self):
        self.weight_gradients = mat_scale(self.weight_gradients, 0)
        self.bias_gradients = mat_scale(self.bias_gradients, 0)


def draw_mnist_digit(data):
    if len(data) == 784:
        for x in range(28):
            print("|", end = "")
            for y in range(28):
                char = data[y + 28 * x]
                if char > 0:
                    print("0  ", end = "")
                else:
                    print("   ", end = "")
            print("|")
        return True
    return False

def num_to_vector(num):
    if num <= 9 and num >= 0:
        vector = cp.zeros(10)
        vector[num] += 1
        return vector.reshape(-1,1)
    return False

def mat_create(data, rows, columns):
    if len(data) == rows * columns:
        mat = cp.array(data.reshape(columns, rows), dtype = float)
    else:
        mat = cp.zeros((columns, rows))
    return mat

def mat_copy(matrix):
    return cp.copy(matrix)

def mat_clear(matrix):
    return cp.zeros(matrix.shape)

def mat_fill(matrix, value):
    mat = cp.array(cp.ones(matrix.shape), dtype = float)
    mat *= value
    return mat

def mat_scale(matrix, value):
    mat = cp.multiply(matrix, value)
    return mat

def mat_sum(matrix):
    return cp.sum(matrix)

def mat_add(matrix_a, matrix_b):
    if (matrix_a.shape != matrix_b.shape):
        return False
    return cp.add(matrix_a, matrix_b)

def mat_sub(matrix_a, matrix_b):
    if (matrix_a.shape != matrix_b.shape):
        return False
    return cp.subtract(matrix_a, matrix_b)

def mat_dot(matrix_a, matrix_b, transpose_a, transpose_b):
    mat_a = mat_copy(matrix_a)
    mat_b = mat_copy(matrix_b)
    if transpose_a:
        mat_a = mat_transpose(mat_a)
    if transpose_b:
        mat_b = mat_transpose(mat_b)

    return cp.dot(mat_a, mat_b)

def mat_mult(matrix_a, matrix_b, transpose_a, transpose_b):
    mat_a = mat_copy(matrix_a)
    mat_b = mat_copy(matrix_b)
    if transpose_a:
        mat_a = mat_transpose(mat_a)
    if transpose_b:
        mat_b = mat_transpose(mat_b)

    if mat_a.shape != mat_b.shape:
        return False
    return cp.multiply(mat_a, mat_b)

def mat_outer(matrix_a, matrix_b, transpose_a, transpose_b):
    mat_a = mat_copy(matrix_a)
    mat_b = mat_copy(matrix_b)
    if transpose_a:
        mat_a = mat_transpose(mat_a)
    if transpose_b:
        mat_b = mat_transpose(mat_b)

    return cp.outer(mat_a, mat_b)

def mat_transpose(matrix):
    mat = cp.transpose(matrix)
    if len(mat.shape) == 1:
        mat = cp.reshape(matrix, (-1,1))
    return mat

def mat_relu(matrix):
    mat = cp.maximum(matrix, 0)
    return mat

def mat_d_relu(matrix):
    return (matrix > 0).astype(float)

def mat_softmax(matrix):
    mat = cp.exp(matrix)
    sum = mat_sum(mat)
    return mat_scale(mat, 1 / sum)

def mat_cross_entropy(matrix_p, matrix_q):
    if (matrix_p.shape != matrix_q.shape):
        return False

    mat = cp.multiply(matrix_p, cp.log(matrix_q))
    return -mat_sum(mat)

def __onStart():
    print("------------------RUNNING------------------")

def __onEnd():
    print("------------------FINISHED-----------------")

main()
