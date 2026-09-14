#Neural Network
#Stoicastic gradient descent
#version 1.5.2

import numpy as np # noqa: I001
#import cupy as cp
import time
import string
from mnist import MNIST

#to do:
#optimize import for data, its taking 30s
#add network exporter and importer
# * save good runs of the exporter to another file (txt)

def main():
    #Gradients don't explode because code is wrong, your just wrong
    #(adjust learning rate)
    time_at_start = time.perf_counter()


    #Importing mnist dataset
    mndata = MNIST('mnist')

    training_images, training_labels = mndata.load_training()
    testing_images, testing_labels = mndata.load_testing()

    input_cache = mndata.process_images_to_numpy(training_images)

    #fitting mnist import to acceptable input data
    # input_cache = np.empty((1,784))
    output_cache = np.empty((1,10))
    for i in range(int(len(training_images))):
    #     input =  np.array([training_images[i]])
        output = np.array([num_to_array(training_labels[i])])
    #     input_cache = np.vstack((input_cache, input), dtype=float)
        output_cache = np.vstack((output_cache, output))
    #     # draw_mnist_digit(training_images[i])
    # input_cache = np.delete(input_cache, 0, axis=0)   
    output_cache = np.delete(output_cache, 0, axis=0)


    # #init network as object
    size = (784,128,10)
    net = NETWORK(size, loss="BINARY_CROSS_ENTROPY_AND_SOFTMAX",activation="SIGMOID")

    
    print(input_cache[690])
    draw_mnist_digit(input_cache[690])
    #train network
    net.feed_optimizer("STOICHASTIC_GRADIENT_DECSENT")
    net.train(time_at_start, input_cache, output_cache, 1, 0.1)


    #test YIPE
    net.test(input_cache, output_cache, telementary=True)

    your_number = 255 * np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 61.2, 12.24, 2.448, 0.49, 0.098, 0.02, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 61.2, 63.24, 63.648, 63.73, 63.746, 63.749, 63.75, 114.75, 186.15, 151.47, 32.742, 7.038, 1.506, 0.321, 0.068, 0.014, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 51, 173.4, 199.92, 205.632, 206.856, 207.117, 207.173, 207.184, 207.187, 217.387, 233.707, 179.035, 42.355, 9.879, 2.277, 0.52, 0.118, 0.026, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 61.2, 148.92, 171.768, 177.48, 178.867, 179.197, 179.274, 179.292, 179.296, 232.337, 246.209, 187.049, 45.881, 11.152, 2.686, 0.641, 0.152, 0.036, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 12.24, 32.232, 40.8, 43.656, 44.505, 44.74, 44.803, 44.819, 95.823, 218.632, 245.968, 137.603, 36.697, 9.57, 2.451, 0.618, 0.154, 0.038, 0.009, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 2.448, 6.936, 9.547, 10.641, 11.029, 11.154, 11.191, 11.202, 123.405, 221.407, 195.475, 66.616, 20.663, 6.046, 1.7, 0.464, 0.124, 0.032, 0.008, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0.49, 1.485, 2.206, 2.569, 2.72, 2.775, 2.793, 53.799, 188.441, 234.97, 137.089, 40.741, 12.281, 3.665, 1.073, 0.307, 0.086, 0.024, 0.006, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0.098, 0.317, 0.505, 0.615, 0.667, 0.688, 0.696, 112.899, 213.268, 191.648, 65.747, 21.298, 6.716, 2.076, 0.63, 0.187, 0.055, 0.016, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.067, 0.114, 0.146, 0.163, 0.17, 51.173, 185.814, 232.816, 135.893, 40.328, 12.325, 3.808, 1.177, 0.361, 0.11, 0.033, 0.01, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0.004, 0.014, 0.026, 0.034, 0.039, 0.042, 112.243, 212.612, 191.086, 65.396, 21.145, 6.694, 2.1, 0.655, 0.203, 0.063, 0.019, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.003, 0.006, 0.008, 0.009, 51.01, 185.651, 232.652, 186.748, 50.429, 14.315, 4.202, 1.26, 0.383, 0.117, 0.036, 0.011, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.002, 0.002, 112.203, 212.571, 242.045, 136.758, 37.437, 10.35, 2.91, 0.834, 0.243, 0.072, 0.022, 0.007, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51.001, 185.641, 232.642, 196.937, 66.739, 20.835, 6.237, 1.83, 0.533, 0.155, 0.045, 0.013, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61.2, 202.368, 240.002, 138.388, 41.025, 12.372, 3.722, 1.11, 0.329, 0.097, 0.028, 0.008, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.24, 206.122, 191.225, 65.923, 21.39, 6.752, 2.095, 0.641, 0.194, 0.058, 0.017, 0.005, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63.648, 155.954, 120.436, 37.272, 11.732, 3.697, 1.158, 0.36, 0.111, 0.034, 0.01, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.73, 33.737, 30.834, 13.621, 5.071, 1.754, 0.582, 0.188, 0.06, 0.019, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.546, 7.257, 7.618, 4.248, 1.864, 0.723, 0.261, 0.09, 0.03, 0.01, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.509, 1.553, 1.834, 1.216, 0.616, 0.268, 0.106, 0.039, 0.014, 0.005, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.102, 0.331, 0.433, 0.33, 0.189, 0.091, 0.039, 0.016, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.07, 0.101, 0.086, 0.055, 0.029, 0.014, 0.006, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004, 0.015, 0.023, 0.022, 0.015, 0.009, 0.005, 0.002, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.003, 0.005, 0.005, 0.004, 0.003, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    your_output = np.array([num_to_array(7)])
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
        print(f"******************Total Cost: {total_cost:.4f}******************")

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
