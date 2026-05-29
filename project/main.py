#Neural Network with GPU and possibly TPU
#version 1.0.1

import numpy as np
import math
#import pygame
import random
#import cupy as cp

def main():
    __onStart()
    
    global a
    a = np.arange(0,10)
    a = mat_create(a, 2, 5)
    print(a)
    a = mat_clear(a)
    print(a)
    a = mat_fill(a, 26)
    print(a)
    a = mat_scale(a, 0.25)
    print(a)
    
    
    
    __onEnd()

def mat_create(data, rows, columns):
    if len(data) == rows * columns:
        mat = np.array(data.reshape(columns, rows), dtype = float)
    else:
        mat = np.zeros((columns, rows))
    return mat

def mat_copy(matrix_out, matrix_in):
    if (matrix_in.shape != matrix_out.shape):
        return False
    
    np.copyto(matrix_out, matrix_in)
    return True

def mat_clear(matrix):
    return np.zeros(matrix.shape)

def mat_fill(matrix, value):
    mat = np.array(np.ones(matrix.shape), dtype = float)
    mat *= value
    return mat

def mat_scale(matrix, value):
    mat = matrix * value
    return mat
    
def mat_sum(matrix):
    return np.sum(matrix)
    
def mat_add(matrix_a, matrix_b):
    if (matrix_a.shape == matrix_b.shape):
        return False
    
    return matrix_a + matrix_b
        

def mat_sub(matrix_a, matrix_b):
    if (matrix_a.shape == matrix_b.shape):
        return False
    
    return matrix_a - matrix_b

def mat_mult(matrix_a, matrix_b, transpose_a, transpose_b):
    nonlocal mat_a
    nonlocal mat_b
    if transpose_a:
        mat_a = matrix_a.T
    else:
        mat_a = matrix_a
    if transpose_b:
        mat_b = matrix_b.T
    else:
        mat_a = matrix_b
    
    if (mat_a.shape != mat_b.shape):
        return False
    
    return np.dot(mat_a, mat_b)

def mat_relu(matrix):
    pass

def mat_softmax(matrix):
    pass

def mat_cross_entropy(matrix_a, matrix_b):
    pass

def __onStart():
    global Network
    Network = []
    print("------------------RUNNING------------------")

def __onEnd():
    pass

main()
