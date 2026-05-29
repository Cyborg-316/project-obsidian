#Neural Network with GPU and possibly TPU
#version 1.0.1

import numpy as np
import math
#import pygame
import random
#import cupy as cp

def main():
    __onStart()
    
    
    
    __onEnd()

def mat_create(data, rows, columns):
    mat = np.array(data.reshape(rows, columns))
    return mat

def mat_copy(matrix_out, matrix_in):
    if(matrix_in.shape != matrix_out.shape):
        return False
    
    np.copyto(matrix_out, matrix_in)
    return True

def mat_clear(matrix):
    pass

def mat_fill(matrix, value):
    pass

def mat_scale(matrix, value):
    pass

def mat_add(matrix_a, matrix_b):
    pass

def mat_sub(matrix_a, matrix_b):
    pass

def mat_mult(matrix_a, matrix_b, transpose_a, transpose_b):
    pass

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
    a = np.arange(0,10)
    print(mat_create(a, 5, 2))

def __onEnd():
    pass

main()
