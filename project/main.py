#Neural Network
#Stoicastic gradient descent
#version 1.5.5.0

import numpy as np # noqa: I001
#import cupy as cp
import time
import string
import sys
from mnist import MNIST

#to do:
#add network exporter and importer
# * save good runs of the exporter to another file (txt)

def main():
    #Gradients don't explode because code is wrong, your just wrong
    #(adjust learning rate)
    time_at_start = time.perf_counter()
    #np.set_printoptions(threshold=np.inf)


    #Importing mnist dataset
    mndata = MNIST('mnist', return_type='numpy')

    training_images, training_labels = mndata.load_training()
    testing_images, testing_labels = mndata.load_testing()

    input_cache = training_images / 255
    output_cache = array_to_array(training_labels)
    test_input_cache = testing_images / 255
    test_output_cache = array_to_array(testing_labels)
    #   [ 0.03279386]] -->  [-0.03881448]]

    #init network as object
    size = (784,128,64,10)
    net = NETWORK(size, loss="BINARY_CROSS_ENTROPY_AND_SOFTMAX",activation="TANH")

    #train network
    net.feed_optimizer("STOICHASTIC_GRADIENT_DECSENT")
    net.train(time_at_start, input_cache, output_cache, 2, .00001)

    #test YIPE
    net.test(test_input_cache, test_output_cache, telementary=True)

    your_number = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 61.2, 63.24, 63.648, 12.73, 2.546, 0.509, 0.102, 0.02, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 112.2, 389.64, 447.168, 459.082, 410.546, 84.655, 17.44, 3.59, 0.738, 0.152, 0.031, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 387.6, 456.96, 526.32, 500.698, 548.956, 497.9, 116.511, 26.79, 6.076, 1.363, 0.303, 0.067, 0.015, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 112.2, 399.84, 514.488, 551.29, 470.522, 245.244, 515.84, 508.748, 125.052, 30.368, 7.289, 1.73, 0.407, 0.095, 0.022, 0.005, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 387.6, 456.96, 528.36, 514.57, 468.172, 187.739, 137.597, 487.687, 505.287, 126.068, 31.287, 7.715, 1.889, 0.459, 0.111, 0.027, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 102, 397.8, 514.08, 551.208, 470.914, 197.097, 133.054, 64.158, 91.351, 472.808, 501.619, 125.537, 31.365, 7.816, 1.941, 0.48, 0.118, 0.029, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 51, 387.6, 514.08, 511.632, 467.568, 187.696, 76.959, 42.002, 72.232, 134.717, 478.505, 502.025, 125.512, 31.375, 7.838, 1.956, 0.487, 0.121, 0.03, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 61.2, 395.76, 436.968, 189.72, 182.458, 125.031, 91.398, 128.68, 397.182, 463.38, 545.377, 464.48, 117.999, 29.875, 7.543, 1.9, 0.477, 0.12, 0.03, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 12.24, 132.6, 164.914, 172.927, 428.077, 467.622, 468.804, 476.497, 531.736, 556.023, 526.28, 198.152, 63.23, 18.621, 5.233, 1.426, 0.381, 0.1, 0.026, 0.007, 0.002, 0, 0, 0, 0, 0, 0, 0, 104.448, 404.41, 470.865, 485.758, 539.767, 558.478, 562.456, 564.791, 576.305, 583.466, 527.949, 196.22, 51.89, 14.102, 3.867, 1.059, 0.288, 0.078, 0.021, 0.005, 0.001, 0, 0, 0, 0, 0, 0, 102, 398.29, 517.54, 554.681, 565.088, 577.971, 584.29, 535.349, 526.028, 526.467, 527.986, 568.187, 458.881, 153.154, 84.451, 17.664, 3.744, 0.806, 0.177, 0.04, 0.009, 0.002, 0, 0, 0, 0, 0, 51, 387.6, 514.178, 563.344, 580.605, 586.139, 589.822, 489.822, 205.034, 146.212, 134.536, 183.504, 456.338, 540.044, 495.64, 422.018, 138.936, 28.536, 5.869, 1.209, 0.25, 0.052, 0.011, 0.002, 0, 0, 0, 0, 61.2, 446.76, 549.188, 579.506, 589.022, 592.032, 491.371, 196.239, 80.255, 45.293, 35.966, 43.894, 151.046, 444.218, 544.972, 550.398, 443.867, 145.481, 30.27, 6.296, 1.309, 0.272, 0.057, 0.012, 0.002, 0.001, 0, 0, 63.24, 459, 558.638, 584.629, 591.73, 491.752, 196.625, 78.573, 31.765, 15.412, 10.276, 10.834, 32.376, 146.319, 444.258, 555.931, 556.96, 446.488, 95.352, 20.329, 4.328, 0.92, 0.195, 0.041, 0.009, 0.002, 0, 0, 63.648, 461.53, 561.033, 586.132, 490.573, 196.465, 78.618, 31.438, 12.641, 5.61, 3.177, 2.802, 7.036, 30.671, 145.986, 446.383, 506.669, 445.631, 108.197, 25.705, 6.007, 1.385, 0.316, 0.072, 0.016, 0.004, 0.001, 0, 63.73, 411.052, 500.417, 472.31, 192.576, 77.808, 31.285, 12.545, 5.037, 2.13, 1.061, 0.773, 1.562, 6.447, 30.486, 95.374, 120.409, 113.208, 44.281, 13.997, 4.001, 1.077, 0.279, 0.07, 0.017, 0.004, 0.001, 0, 12.746, 84.76, 117.035, 117.869, 62.089, 27.979, 11.853, 4.88, 1.983, 0.823, 0.377, 0.23, 0.358, 1.361, 6.369, 20.349, 28.151, 28.272, 14.511, 5.702, 1.94, 0.604, 0.176, 0.049, 0.013, 0.003, 0.001, 0, 2.549, 17.462, 26.899, 28.954, 18.209, 9.238, 4.218, 1.82, 0.761, 0.317, 0.139, 0.074, 0.086, 0.289, 1.332, 4.336, 6.498, 6.954, 4.293, 1.999, 0.788, 0.278, 0.091, 0.028, 0.008, 0.002, 0.001, 0, 0.51, 3.594, 6.099, 7.01, 5.044, 2.856, 1.415, 0.647, 0.281, 0.12, 0.052, 0.025, 0.022, 0.062, 0.279, 0.923, 1.484, 1.688, 1.196, 0.639, 0.285, 0.113, 0.041, 0.014, 0.004, 0.001, 0, 0, 0.102, 0.739, 1.368, 1.676, 1.344, 0.84, 0.451, 0.22, 0.1, 0.044, 0.019, 0.009, 0.006, 0.014, 0.059, 0.196, 0.336, 0.405, 0.32, 0.192, 0.095, 0.042, 0.016, 0.006, 0.002, 0.001, 0, 0, 0.02, 0.152, 0.304, 0.396, 0.348, 0.238, 0.138, 0.071, 0.034, 0.016, 0.007, 0.003, 0.002, 0.003, 0.012, 0.042, 0.076, 0.096, 0.083, 0.055, 0.03, 0.014, 0.006, 0.002, 0.001, 0, 0, 0, 0.004, 0.031, 0.067, 0.093, 0.088, 0.065, 0.041, 0.022, 0.011, 0.005, 0.002, 0.001, 0.001, 0.001, 0.003, 0.009, 0.017, 0.023, 0.021, 0.015, 0.009, 0.005, 0.002, 0.001, 0, 0, 0, 0, 0.001, 0.006, 0.015, 0.021, 0.022, 0.017, 0.012, 0.007, 0.004, 0.002, 0.001, 0, 0, 0, 0.001, 0.002, 0.004, 0.005, 0.005, 0.004, 0.003, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0.001, 0.003, 0.005, 0.005, 0.005, 0.003, 0.002, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.001, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    your_number = your_number / np.max(your_number)
    your_output = np.array([num_to_array(2)])
    net.test(your_number, your_output, telementary=True)

    # net.desmos_format1D()
    #EXPORTER.export_parameters(net, "model_archive/model_01")
    #EXPORTER.import_parameters("model_archive/model_01")

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), loss="MEAN_SQUARED_ERROR", activation="NONE", IAFLL=False):
        self.size = size
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

class EXPORTER:
    def export_parameters(net, file_name):
        condition = np.array([str(net.size),str(net.loss),str(net.activation_function),str(net.Include_activation_for_last)])
        np.save(file_name, condition)
        for layer in net.Layers:
            if layer.type == "Dense":
                np.savez(file_name, layer.weights, layer.biases)

        print("Export complete!!!!")

    def import_parameters(file_name):
        #same format as export parameters
        #overwrite prev layer parameters with imported ones
        condition = np.load(f"{file_name}.npy")
        print(condition)    
        parameters = np.load(f"{file_name}.npz")
        print(parameters)

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
            dpda = eq + np.sum(uneq, axis=1).reshape(-1,1)
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
