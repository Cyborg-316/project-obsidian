#Neural Network with GPU and possibly TPU
#version 1.0.2

import numpy as np
import math
#import pygame
import random
import cupy as cp
from mnist import MNIST

def main():
    __onStart()

    mndata = MNIST('mnist')
    
    images, labels = mndata.load_training()
    train_images = images[:50000]
    train_labels = labels[:50000]
    test_images = images[50000:]
    test_labels = labels[50000:]

    draw_mnist_digit(train_images[7])
    print(train_labels[7])
    #print(num_to_vector(train_labels[7]))

    #Layer1 = structure(784, 128)
    #Laeyr2 = structure(128, 10)
    #128 neurons in 1 hideen layer with 784 inputs and 10 outputs
    
    # inputs = cp.arange(0,10, 1, dtype = float)
    # outputs = cp.arange(0,20, 2, dtype = float) + 1
    # print(inputs, "\n", outputs)

    Layer1 = structure(784, 10)

    print(mat_softmax(Layer1.activations(mat_scale(cp.array([train_images[0]]), 1/255))))
        

    
    
    __onEnd()

class structure:
    def __init__(self, input_neurons, output_neurons):
        self.weights = cp.ones((output_neurons, input_neurons))
        self.biases = cp.ones(output_neurons)

    def activations(self, inputs):   
        return mat_relu(self.weighted_sums(inputs))
    
    def weighted_sums(self, inputs):
        z = mat_mult(self.weights, cp.array([inputs]), False, False) + self.biases
        return z
    
    def gradients():
        pass





def draw_mnist_digit(data):
    if len(data) == 784:
        for x in range(28):
            print("|", end = "")
            for y in range(28):
                char = data[y + 28 * x]
                if char > 0:
                    print("0", end = "")
                else:
                    print(" ", end = "")
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
    if (matrix_a.shape == matrix_b.shape):
        return False
    
    return cp.add(matrix_a, matrix_b)
        

def mat_sub(matrix_a, matrix_b):
    if (matrix_a.shape == matrix_b.shape):
        return False
    
    return cp.subtract(matrix_a, matrix_b)

def mat_mult(matrix_a, matrix_b, transpose_a, transpose_b):
    mat_a = mat_copy(matrix_a)
    mat_b = mat_copy(matrix_b)
    if transpose_a:
        mat_a = cp.transpose(mat_a)
    if transpose_b:
        mat_b = cp.transpose(mat_b)
    
    if len(mat_a.shape) == 1 and len(mat_b.shape) == 1 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 1 and len(mat_b.shape) == 2 and mat_a.shape[0] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 1 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    elif len(mat_a.shape) == 2 and len(mat_b.shape) == 2 and mat_a.shape[1] == mat_b.shape[0]:
        return cp.dot(mat_a, mat_b)
    return False
    
def mat_transpose(matrix):
    mat = cp.transpose(matrix)
    return mat

def mat_relu(matrix):
    mat = cp.maximum(matrix, 0)
    return mat

def mat_softmax(matrix):
    mat = cp.exp(matrix)
    sum = mat_sum(mat)
    return mat_scale(mat, 1 / sum)

def mat_cross_entropy(matrix_p, matrix_q):
    if (matrix_p.shape != matrix_q.shape):
        False
    
    mat_p = mat_copy(matrix_p)
    mat_q = mat_copy(matrix_q)
    mat = cp.multiply(mat_p, cp.log(mat_q))
    return -mat_sum(mat)


def __onStart():
    print("------------------RUNNING------------------")

def __onEnd():
    print("------------------FINISHED-----------------")

main()
