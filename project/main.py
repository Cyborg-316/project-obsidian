#Neural Network
#Stoicastic gradient descent
#version 1.5.3

import numpy as np # noqa: I001
#import cupy as cp
import time
import string
from mnist import MNIST

#to do:
#add network exporter and importer
# * save good runs of the exporter to another file (txt)

def main():
    #Gradients don't explode because code is wrong, your just wrong
    #(adjust learning rate)
    time_at_start = time.perf_counter()


    #Importing mnist dataset
    mndata = MNIST('mnist', return_type='numpy')

    training_images, training_labels = mndata.load_training()
    testing_images, testing_labels = mndata.load_testing()

    input_cache = training_images
    output_cache = array_to_array(training_labels)
    test_input_cache = testing_images
    test_output_cache = array_to_array(testing_labels)


    #init network as object
    size = (784,128,64,32,10)
    net = NETWORK(size, loss="BINARY_CROSS_ENTROPY_AND_SOFTMAX",activation="TANH")

    #train network
    net.feed_optimizer("STOICHASTIC_GRADIENT_DECSENT")
    net.train(time_at_start, input_cache, output_cache, 1, 0.05)

    #test YIPE
    net.test(test_input_cache, test_output_cache, telementary=True)

    your_number = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 61.2, 63.24, 12.648, 2.53, 0.506, 0.101, 0.02, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 377.4, 444.72, 407.592, 84.048, 17.316, 3.564, 0.733, 0.151, 0.031, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61.2, 444.72, 534.888, 494.496, 115.709, 26.605, 6.034, 1.353, 0.301, 0.066, 0.015, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.24, 458.592, 555.696, 516.038, 126.349, 30.591, 7.325, 1.736, 0.407, 0.095, 0.022, 0.005, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.648, 461.448, 560.429, 521.293, 129.529, 32.024, 7.87, 1.921, 0.466, 0.112, 0.027, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.73, 462.036, 561.493, 522.557, 130.417, 32.488, 8.072, 1.999, 0.493, 0.121, 0.03, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.746, 411.156, 551.53, 520.817, 130.247, 32.547, 8.124, 2.024, 0.503, 0.125, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.749, 135.781, 494.462, 509.056, 127.861, 32.082, 8.041, 2.013, 0.503, 0.126, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.55, 78.666, 471.626, 502.136, 125.999, 31.616, 7.931, 1.989, 0.498, 0.125, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.51, 66.835, 464.692, 499.366, 125.073, 31.338, 7.854, 1.969, 0.493, 0.124, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.102, 64.387, 462.816, 498.436, 124.702, 31.208, 7.812, 1.956, 0.49, 0.123, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 63.882, 462.339, 498.155, 124.571, 31.156, 7.794, 1.95, 0.488, 0.122, 0.031, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004, 63.777, 462.223, 498.076, 124.529, 31.137, 7.786, 1.947, 0.487, 0.122, 0.03, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 63.756, 462.196, 498.054, 124.517, 31.131, 7.783, 1.946, 0.487, 0.122, 0.03, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.751, 462.189, 498.049, 124.513, 31.129, 7.782, 1.946, 0.486, 0.122, 0.03, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.75, 462.188, 498.047, 124.512, 31.128, 7.782, 1.946, 0.486, 0.122, 0.03, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.75, 411.188, 436.847, 112.272, 28.68, 7.292, 1.848, 0.467, 0.118, 0.03, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.75, 84.788, 104.327, 43.32, 14.4, 4.338, 1.237, 0.341, 0.092, 0.024, 0.006, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.55, 17.468, 24.359, 13.536, 5.587, 1.985, 0.644, 0.197, 0.058, 0.016, 0.005, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.51, 3.596, 5.591, 3.825, 1.882, 0.774, 0.284, 0.096, 0.031, 0.009, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.102, 0.74, 1.266, 1.018, 0.58, 0.271, 0.111, 0.041, 0.014, 0.005, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.152, 0.284, 0.26, 0.168, 0.088, 0.04, 0.016, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004, 0.031, 0.063, 0.065, 0.047, 0.027, 0.013, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0]], dtype=float)
    your_number = your_number / np.max(your_number) * 255
    your_output = np.array([num_to_array(2)])
    net.test(your_number, your_output, telementary=True)

    # net.desmos_format1D()

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), loss="MEAN_SQUARED_ERROR", activation="NONE", IAFLL=False):
        self.loss = loss
        self.Layers = architecture(size, loss, activation)
        self.activation_function = activation
        self.Include_activation_for_last = IAFLL

    def train(self, time_at_start, input_cache, output_cache, epochs=1000, lr=0.05):
        print(f"---------TRAINING-AT--{time_at_start - time.perf_counter():.6f}---------")
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
            return
        elif self.optimizer == "MOMENTUM":
            #momentum gradient descent here
            return
        else:
            print("Please enter an optimizer for training")
            return
        print(f"-----FINISHED-TRAINING-AT--{time_at_start - time.perf_counter():.6f}----")

    def foward_pass(self, inputs, outputs):
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)

        if self.Include_activation_for_last:
            self.Layers[-1].foward(self.Layers[-2].outputs, outputs)
        else:
            self.Layers[-1].foward(self.Layers[-2].weighted_sums, outputs)

    def foward_return(self, inputs, outputs):
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)

        predicted = None
        #insert softmax logic here
        if self.Include_activation_for_last:
            predicted = self.Layers[-2].outputs
        else:
            predicted = self.Layers[-2].weighted_sums
        
        cost = self.Layers[-1].foward(predicted, outputs)
        predicted = self.Layers[-1].predicted
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
                layer.backprop(following_layer, self.Include_activation_for_last)
                following_layer = layer

    def update_parameters(self, lr):
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                layer.update_parameters(lr)

    def test(self, input_cache, output_cache, telementary=True):
        print("\nTEST:")
        total_cost = 0
        best_case_index = 0
        worst_case_index = 0
        best_case_cost = 1000
        worst_case_cost = 1000
        for index, inputs in enumerate(input_cache):
            outputs = output_cache[index]
            cost, predicted = self.foward_return(inputs, outputs)
            total_cost += cost

            if index == 0:
                best_case_cost = cost
                worst_case_cost = cost
            if worst_case_cost < cost:
                worst_case_index = index
                worst_case_cost = cost
            if best_case_cost > cost:
                best_case_index = index
                best_case_cost = cost

        if worst_case_index == best_case_index:
            if telementary:
                outputs = output_cache[worst_case_index]
                inputs = input_cache[worst_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Predicted:\n", predicted)
                print("Actual:\n", outputs.reshape(-1,1))

                draw_mnist_digit(input_cache[best_case_index])
        else:
            if telementary:
                outputs = output_cache[worst_case_index]
                inputs = input_cache[worst_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Worst Case Predicted:\n", predicted)
                print("Worst Case Actual:\n", outputs.reshape(-1,1))

                draw_mnist_digit(input_cache[worst_case_index])

                outputs = output_cache[best_case_index]
                inputs = input_cache[best_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Best Case Predicted:\n", predicted)
                print("Best Case Actual:\n", outputs.reshape(-1,1))

                draw_mnist_digit(input_cache[best_case_index])

            print(f"Worst Case: {worst_case_cost:.6f}\nBest Case: {best_case_cost:.6f}")
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

        print("\nDESMOS COPY PASTE BELOW:", end="")

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
                for i, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)
                    m = weight[0]
                    b = layer.biases[i][0]

                    
                    if b >= 0:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}+{b:.5f})")
                    else:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}{b:.5f})")
                prev_func.append(temp)
            elif not layer.type == "Cost":
                for i, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)

                    b = layer.biases[i][0]
                    equation = ""
                    for i, m in enumerate(weight):
                        if m >= 0:
                            equation += f"+{m:.5f}{prev_func[index - 1][i]}({input})"
                        else:
                            equation += f"{m:.5f}{prev_func[index - 1][i]}({input})"
                    if b >= 0:
                        if index == len(self.Layers) - 2 and not self.Include_activation_for_last:
                            print(f"{func[-1]}({input})={equation}+{b:.5f}")
                        else:
                            print(f"{func[-1]}({input})={func[0]}({equation}+{b:.5f})")
                    else:
                        if index == len(self.Layers) - 2 and not self.Include_activation_for_last:
                            print(f"{func[-1]}({input})={equation}{b:.5f}")
                        else:
                            print(f"{func[-1]}({input})={func[0]}({equation}{b:.5f})")
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
            a =  1 / (1 + np.exp(-np.maximum(np.minimum(z,15),-15)))
        elif self.activation == "TANH":
            a = np.tanh(z)
        elif self.activation == "NONE":
            a = z
        self.weighted_sums = z
        self.outputs = a
        return a

    def telementary_parameters(self):
        print("\nweights: ")
        print(self.weights)
        print("\nbiases: ")
        print(self.biases)

    def backprop(self, following_layer, IAFLL):
        aft_deltas = following_layer.deltas
        #update
        # partial_derivative = None
        if self.activation == "RELU":
            partial_derivative = (self.weighted_sums > 0).astype(dtype=float)
        elif self.activation == "SIGMOID":
            partial_derivative = self.outputs * (1 - self.outputs)
        elif self.activation == "TANH":
            partial_derivative = 1 - np.tanh(self.weighted_sums) ** 2
        elif self.activation == "NONE":
            partial_derivative = 1

        if not following_layer.type == "Cost":
            aft_weights = following_layer.weights
            self.deltas = np.transpose(aft_weights) @ aft_deltas * partial_derivative     
        else:
            if IAFLL:
                self.deltas = aft_deltas * partial_derivative
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

    def foward(self, inputs, actual):
        self.predicted = inputs
        c = None
        if self.cost == "MEAN_SQUARED_ERROR":
            c = np.sum((actual.reshape(-1,1) - inputs) ** 2)
        elif self.cost == "BINARY_CROSS_ENTROPY":
            c = binary_cross_entropy(inputs, actual.reshape(-1,1))
        elif self.cost == "BINARY_CROSS_ENTROPY_AND_SOFTMAX":
            self.predicted = stable_softmax(inputs)
            c = binary_cross_entropy(stable_softmax(inputs), actual.reshape(-1,1))
        self.outputs = c
        return c

    def backprop(self, actual):
        if self.cost == "MEAN_SQUARED_ERROR":
            self.deltas = 2 * (self.predicted - actual.reshape(-1,1))
        elif self.cost == "BINARY_CROSS_ENTROPY":
            self.deltas = -1 * (actual.reshape(-1,1) / self.predicted - (1-actual.reshape(-1,1))/(1-self.predicted))
        elif self.cost == "BINARY_CROSS_ENTROPY_AND_SOFTMAX":
            # A=np.arange(16).reshape(4,4)
            # print(A)
            # print(A * (-np.eye(4,4) + 1))
            # print(A * (-np.eye(4,4) + 1) + np.identity(4))
            sfmx = stable_softmax(self.predicted)
            dcdp = -(actual.reshape(-1,1) / sfmx - (1 - actual.reshape(-1,1))/(1 - sfmx))
            eq = sfmx * (1 - sfmx)      
            uneq = (sfmx @ np.transpose(sfmx)) * (-np.eye(sfmx.size) + 1) + np.eye(sfmx.size)  
            uneq = np.prod(uneq, axis=1).reshape(sfmx.size, 1)
            dpda = eq * uneq
            self.deltas = dcdp * dpda
        
def num_to_array(integer):
    array = np.zeros(10)
    array[integer] += 1
    return array

def array_to_array(array):
    return np.eye(10)[array]

def architecture(size, loss, activation_type):
    Layers = []
    for index in range(len(size) - 1):
        Layers.append(DENSE_LAYER(size[index], size[index + 1], activation_type))
    Layers.append(COST_LAYER(loss))
    return Layers

def stable_softmax(matrix):
    shiftx = matrix - np.max(matrix)
    mat = np.exp(shiftx)
    return mat / np.sum(mat)

def cross_entropy(matrix_p, matrix_q):
    if not matrix_p.shape == matrix_q.shape:
        print("Cross Entopy shape mismatch")
        return False

    mat = np.multiply(matrix_p, np.log(matrix_q))
    return np.minimum(-np.sum(mat), 1000)

def binary_cross_entropy(matrix_p, matrix_q):
    if not matrix_p.shape == matrix_q.shape:
        print("Cross Entopy shape mismatch")
        return False

    mat = matrix_q * np.log(matrix_p) + (1 - matrix_q) * np.log((1 - matrix_p))
    return np.minimum(-np.sum(mat), 1000)

def draw_mnist_digit(image_list):
    for x in range(28):
        print("", end = "")
        for y in range(28):
            char = image_list[y + 28 * x] / max(image_list) * 4
            if char < 1:
                col_char = "░░"
            elif char < 2:
                col_char = "▒▒"
            elif char < 3:
                col_char = "▓▓"
            else:
                col_char = "██"

            print(f"{col_char}", end="")
        print("")

start = time.perf_counter()
print("\n---------------RUNNING---------------\n")
main()
end = time.perf_counter()
print(f"\n-----FINISHED--IN--{(end - start):.9f}s------\n")
