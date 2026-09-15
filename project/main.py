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
    print("iran")

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
    net.train(time_at_start, input_cache, output_cache, 2, .000005)

    #test YIPE
    net.test(test_input_cache, test_output_cache, telementary=True)

    your_number = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25.5, 28.05, 2.805, 0.281, 0.028, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 313.65, 314.67, 57.248, 31.253, 3.128, 0.313, 0.031, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 316.2, 368.985, 374.366, 349.161, 318.541, 57.667, 31.298, 3.133, 0.314, 0.031, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 316.2, 369.24, 379.823, 355.919, 351.008, 372.955, 349.062, 318.536, 32.167, 3.248, 0.328, 0.033, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 316.2, 369.24, 354.348, 328.417, 68.434, 67.444, 324.54, 373.36, 349.69, 38.186, 4.143, 0.447, 0.048, 0.005, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25.5, 313.65, 368.985, 328.823, 68.317, 39.673, 10.811, 7.825, 58.737, 349.21, 350.39, 38.858, 4.3, 0.475, 0.052, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.05, 340.17, 351.416, 93.524, 16.184, 5.586, 1.64, 0.947, 31.468, 344.068, 349.946, 38.88, 4.318, 0.479, 0.053, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.305, 342.848, 375.426, 327.395, 59.858, 6.544, 0.818, 0.176, 28.664, 343.273, 349.822, 38.87, 4.319, 0.48, 0.053, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.331, 317.618, 375.304, 376.27, 324.113, 58.566, 5.938, 0.611, 53.928, 345.72, 350.054, 38.892, 4.321, 0.48, 0.053, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.833, 57.545, 323.785, 376.005, 376.012, 323.958, 58.49, 56.91, 317.084, 372.28, 327.233, 36.613, 4.093, 0.457, 0.051, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.283, 5.783, 58.457, 323.946, 375.996, 375.995, 349.448, 346.636, 372.372, 354.965, 68.22, 10.483, 1.458, 0.192, 0.024, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.028, 0.581, 5.904, 83.985, 351.998, 378.799, 378.825, 378.546, 381.092, 328.606, 39.683, 5.017, 0.647, 0.084, 0.011, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003, 0.058, 51.596, 319.558, 373.156, 381.195, 382.002, 382.055, 356.815, 68.542, 10.822, 1.584, 0.223, 0.031, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25.506, 313.71, 369.327, 354.748, 354.094, 354.11, 379.616, 354.143, 42.269, 5.309, 0.689, 0.091, 0.012, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.051, 340.176, 351.45, 70.62, 42.471, 90.658, 353.027, 351.217, 39.349, 4.466, 0.516, 0.061, 0.007, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.305, 342.848, 349.93, 67.555, 62.003, 321.266, 373.429, 327.465, 36.681, 4.115, 0.463, 0.052, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28.331, 317.618, 372.755, 350.031, 347.203, 372.847, 355.128, 68.259, 10.494, 1.461, 0.192, 0.024, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.833, 57.545, 323.53, 347.856, 350.006, 352.785, 325.791, 39.405, 4.99, 0.645, 0.084, 0.011, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.283, 5.783, 32.931, 38.079, 38.808, 39.159, 36.495, 7.59, 1.258, 0.19, 0.027, 0.004, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.028, 0.581, 3.351, 4.143, 4.295, 4.345, 4.084, 1.167, 0.243, 0.043, 0.007, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003, 0.058, 0.341, 0.448, 0.474, 0.482, 0.457, 0.162, 0.04, 0.008, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006, 0.035, 0.048, 0.052, 0.053, 0.051, 0.021, 0.006, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.004, 0.005, 0.006, 0.006, 0.006, 0.003, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    your_number = your_number / np.max(your_number)
    your_output = np.array([num_to_array(7)])
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
