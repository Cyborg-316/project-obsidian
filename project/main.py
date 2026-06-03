#Neural Network with GPU and possibly TPU
#version 1.0.5

#import numpy as cp
import math
#import pygame
import random
import cupy as cp
# from mnist import MNIST

def main():
    __onStart()

    # mndata = MNIST('mnist')
    
    # images, labels = mndata.load_training()
    # train_images = images[:50000]
    # train_labels = labels[:50000]
    # test_images = images[50000:]
    # test_labels = labels[50000:]

    # draw_mnist_digit(train_images[102])
    # print(train_labels[102])
    #print(num_to_vector(train_labels[7]))

    #Layer1 = structure(784, 128)
    #Layer2 = structure(128, 10)
    #128 neurons in 1 hideen layer with 784 inputs and 10 outputs
    
    inputs = cp.arange(0,10, 1, dtype = float)
    outputs = cp.arange(0,20, 2, dtype = float) -3


    Layer1 = structure(1, 1, "last_layer")

    for epoch in range(1000):
        tot_weight_grad = cp.zeros(Layer1.weights.shape)
        tot_bias_grad = cp.zeros(Layer1.biases.shape)
        for i in range(len(inputs)):
            Layer1.input(inputs[i])
            Layer1.expected(outputs[i])
            tot_bias_grad += Layer1.bias_gradients()
            tot_weight_grad += Layer1.weight_gradients()
        Layer1.weights += tot_weight_grad
        Layer1.biases += tot_bias_grad

    for i in range(len(inputs)):
        Layer1.input(inputs[i])
        Layer1.expected(outputs[i])
        print(inputs[i], outputs[i], "     pred:", Layer1.activations())
            

    print(Layer1.weights, "x + ", Layer1.biases)

    #REMEBER TO FEED IT THE SOFT MAX VERSIONS FOR CROSS ENTROPY
    __onEnd()

class structure:
    def __init__(self, num_input_neurons, num_output_neurons, layer_type):
        self.weights = cp.ones((num_output_neurons, num_input_neurons))
        self.biases = cp.ones(num_output_neurons)
        self.type = layer_type
        
    def input(self, inputs):
        self.inputs = cp.array([inputs])
        
    def expected(self, outputs):
        self.actual = cp.array([outputs])

    def weighted_sums(self):
        z = mat_dot(self.weights, self.inputs, False, False) + self.biases
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
        if self.type == "last_layer":
            #last layer gradients no outputs needed
            mat = 2 * (mat_sub(self.weighted_sums(), self.actual))
            return mat
        elif self.type == "hidden_layer":
            #any layer gradients outputs needed
            print("I am a hidden layer")

    def weight_gradients(self):
        mat = mat_mult(self.deltas(), mat_transpose(self.inputs), False, False)
        #print(self.deltas(inputs, outputs), prev_activations)
        #print( mat_mult(self.deltas(inputs, outputs), prev_activations, False, False))
        return -learning_rate * mat
        
    def bias_gradients(self):
        return -learning_rate * self.deltas()

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
    
    if len(mat_a.shape) == 1 and len(mat_b.shape) == 1 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 1 and len(mat_b.shape) == 2 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 1 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 2 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    return False

def mat_mult(matrix_a, matrix_b, transpose_a, transpose_b):
    mat_a = mat_copy(matrix_a)
    mat_b = mat_copy(matrix_b)
    if transpose_a:
        mat_a = mat_transpose(mat_a)
    if transpose_b:
        mat_b = mat_transpose(mat_b)
    
    if len(mat_a.shape) == 1 and len(mat_b.shape) == 1 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.outer(mat_a, mat_b)
    elif len(mat_a.shape) == 1 and len(mat_b.shape) == 2 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.outer(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 1 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.outer(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 2 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.outer(mat_a, mat_b)
    return False
    
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
    global learning_rate
    learning_rate = 0.001
    print("------------------RUNNING------------------")

def __onEnd():
    print("------------------FINISHED-----------------")

main()