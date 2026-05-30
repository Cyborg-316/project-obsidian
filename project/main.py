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

    mndata = MNIST(r"C:\Users\Anton\OneDrive\Documents\Repositories\project-obsidian\mnist")
    
    images, labels = mndata.load_training()
    train_images = images[:50000]
    train_labels = labels[:50000]
    test_images = images[50000:]
    test_labels = labels[50000:]

    draw_mnist_digit(train_images[800])
    print(num_to_vector(train_labels[800]))
    
    
    __onEnd()

def init_parameters(numInputs, numOutputs, numNuerons, numHiddenLayers):
    W1 = cp.random.randn(numNuerons, numInputs)
    b1 = cp.random.randn(numNuerons, 1)
    W2 = cp.random.randn(numNuerons, numNuerons)
    b2 = cp.random.randn(numNuerons, 1)
    return W1, b1, W2, b2

def foward_prop(W1, b1, W2, b2, x):
    Z1 = mat_add(mat_mult(W1, x, False, False), b1)
    A1 = mat_relu(Z1)
    Z2 = mat_add(mat_mult(W2, A1, False, False), b2)
    A2 = mat_softmax(Z2)
    return Z1, A1, Z2, A2

def back_propr(Z1, A1, Z2, A2):
    pass

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
    
def mat_sum(matrix, axis):
    return cp.sum(matrix, axis=axis)
    
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
    


def mat_relu(matrix):
    mat = cp.maximum(matrix, 0)
    return mat

def mat_softmax(matrix):
    mat = cp.exp(matrix)
    sum = mat_sum(mat, 1)
    return mat_scale(mat, 1 / sum)

def mat_cross_entropy(matrix_p, matrix_q):
    if (matrix_p.shape != matrix_q.shape):
        False
    
    mat_p = mat_copy(matrix_p)
    mat_q = mat_copy(matrix_q)
    mat = cp.multiply(mat_p, cp.log(mat_q))
    return -mat_sum(mat, None)


def __onStart():
    print("------------------RUNNING------------------")

def __onEnd():
    print("------------------FINISHED-----------------")

main()
