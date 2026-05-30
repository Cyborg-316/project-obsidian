#Neural Network with GPU and possibly TPU
#version 1.0.1

import numpy as np
import math
#import pygame
import random
import cupy as cp

def main():
    __onStart()
    
    y_true = cp.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
    ])

    y_pred = cp.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    print(mat_cross_entropy(y_true, y_pred))
    
    
    __onEnd()

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
    
    return matrix_a + matrix_b
        

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
    global Network
    Network = []
    print("------------------RUNNING------------------")

def __onEnd():
    print("------------------FINISHED-----------------")

main()
