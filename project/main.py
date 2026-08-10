#Neural Network
#Stoicastic gradient descent
#version 1.4.6

import numpy as np # noqa: I001
import time
import string
from mnist import MNIST

def main():
    size = (2,6,1)
    net = NETWORK(size, activation="TANH")

    
    input_cache = np.array([[0,0],[0,1],[1,0],[1,1]])
    output_cache = np.array([[1],[0],[0],[0]])
    print(input_cache, "\n", output_cache)
    # input_cache = normalize(input_cache)
    # output_cache = normalize(output_cache)

    net.feed_optimizer("STOICHASTIC_GRADIENT_DECSENT")
    # net.telementary()
    net.train(input_cache, output_cache, 10000)

    net.telementary()
    net.test(input_cache, output_cache)

    # net.desmos_format1D()

    #y = 5x -2 
    

    mndata = MNIST('mnist')

    # training_images, training_labels = mndata.load_training()
    # testing_images, testing_labels = mndata.load_testing()

    # print(len(training_images), len(training_labels))
    # print(len(testing_images), len(testing_labels))

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), loss="MEAN_SQUARED_ERROR", activation="NONE"):
        self.loss = loss
        self.Layers = architecture(size, loss, activation)
        self.activation_function = activation

    def train(self, input_cache, output_cache, epochs=1000, lr=0.01):
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
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)

        self.Layers[-1].foward(self.Layers[-2].weighted_sums, outputs)

    def foward_return(self, inputs, outputs):
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)
        predicted = self.Layers[-2].weighted_sums
        
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
        print("\nTEST:")
        total_cost = 0
        for index, inputs in enumerate(input_cache):
            outputs = output_cache[index]
            cost, predicted = self.foward_return(inputs, outputs)
            total_cost += cost

            try:
                predicted[0][0]
            except:
                print(f"Predicted: {predicted[0]:.6f}    Actual: {outputs[0]:.6f}    Cost: {cost:.6f}")
            else:
                print(f"Predicted: {predicted[0][0]:.4f}    Actual: {outputs[0]:.6f}    Cost: {cost:.6f}")
        print(f"Total Cost: {total_cost:.4f}")

    def feed_optimizer(self, type):
        if type == "STOICHASTIC_GRADIENT_DECSENT":
            self.optimizer = type
        elif type == "GRADIENT_DECSENT":
            self.optimizer = type
        elif type == "MOMENTUM":
            self.optimizer = type

    def desmos_format1D(self, input = "x"):
        if not self.Layers[0].num_inputs == 1:
            print("\n******Desmos 1D only works with one input******")
            return

        abc = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        abc.remove(input)
        abc.remove("e")
        
        if self.number_of_nuerons() > len(abc):
            print("\n******Desmos has too little functions to represent network******")
            return

        func = []
        prev_func = []

        func.append(abc[0])
        abc.pop(0)
        if self.activation_function == "NONE":
            print(f"\n{func[0]}({input})={input}")
        elif self.activation_function == "RELU":
            print(f"\n{func[0]}({input})=max({input},0)")
        elif self.activation_function == "SIGMOID":
            print(f"\n{func[0]}({input})=1/(1+exp(-{input}))")
        elif self.activation_function == "TANH":
            print(f"\n{func[0]}({input})=tanh({input})")
        
        for index, layer in enumerate(self.Layers):
            temp = []
            if (not layer.type == "Cost") and (index == 0):
                for index, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)
                    m = weight[0]
                    b = layer.biases[index][0]

                    if b >= 0:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}+{b:.5f})")
                    else:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}{b:.5f})")
                prev_func.append(temp)
            elif not layer.type == "Cost":
                for index, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)

                    b = layer.biases[index][0]
                    equation = ""
                    for i, m in enumerate(weight):
                        if m >= 0:
                            equation += f"+{m:.5f}{prev_func[index - 1][i]}({input})"
                        else:
                            equation += f"{m:.5f}{prev_func[index - 1][i]}({input})"
                    if b >= 0:
                        print(f"{func[-1]}({input})={equation}+{b:.5f}")
                    else:
                        print(f"{func[-1]}({input})={equation}{b:.5f}")
                prev_func.append(temp)

    def number_of_nuerons(self):
        num_nuerons = 0
        for layer in self.Layers:
            num_nuerons += layer.num_outputs
        return num_nuerons

class DENSE_LAYER:
    def __init__(self, num_inputs, num_outputs, activation_type):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_outputs, num_inputs) * np.sqrt(2 / num_inputs)
        self.biases = np.random.randn(num_outputs, 1) * np.sqrt(2 / num_inputs)
        self.activation = activation_type
        self.type = "Dense"

    def foward(self, inputs):
        self.inputs = inputs
        z = self.weights @ inputs
        z += self.biases
        a = None
        if self.activation == "RELU":
            a = np.maximum(z, 0)
        elif self.activation == "SIGMOID":
            a =  1 / (1 + np.exp(-z))
        elif self.activation == "TANH":
            a = np.tanh(z)
        elif self.activation == "NONE":
            a = z
        self.weighted_sums = z
        self.outputs = a
        return a

    def telementary_parameters(self):
        print("\nweights: ", end="")
        print(self.weights)
        print("\nbiases: ", end="")
        print(self.biases)

    def backprop(self, following_layer):
        aft_deltas = following_layer.deltas
        #update
        if not following_layer.type == "Cost":
            partial_derivative = None
            if self.activation == "RELU":
                partial_derivative = (self.weighted_sums > 0).astype(dtype=float)
            elif self.activation == "SIGMOID":
                partial_derivative = self.outputs * ( 1 - self.outputs)
            elif self.activation == "TANH":
                partial_derivative = 1 - np.tanh(self.weighted_sums) ** 2
            elif self.activation == "NONE":
                partial_derivative = 1

            aft_weights = following_layer.weights
            self.deltas = np.transpose(aft_weights) @ aft_deltas * partial_derivative
        else:
            self.deltas = aft_deltas
        
    def update_parameters(self, lr):
        self.biases = self.biases - lr * self.deltas
        self.weights = self.weights - lr * (self.deltas @ self.inputs.T)

class COST_LAYER:
    def __init__(self, cost_type):
        self.cost = cost_type
        self.type = "Cost"
        self.num_outputs = 0

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

def array_extender(array):
    new_array = np.array([[array[0]]])
    for index, i in enumerate(array):
        if not index == 0:
            new_array = np.concatenate((new_array,np.array([[i]])))
    return new_array

def normalize(array):
    return array / np.max(array)

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
