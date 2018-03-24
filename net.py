import numpy as np
from random import random
from math import exp

class N():

    def __init__(self, numI, numH, numO):
        self.inodes = numI
        self.hnodes = numH
        self.onodes = numO

        self.w_IH = matrix(random, self.hnodes, self.inodes)
        self.w_HO = matrix(random, self.onodes, self.hnodes)

        self.bias_H = matrix(random, self.hnodes, 1)
        self.bias_O = matrix(random, self.onodes, 1)

        self.learning_rate = 0.1

    #Recibe las entradas en formato de lista
    def predict(self, inputs_array):
        self.inputs = matrix(inputs_array, len(inputs_array), 1)
        #Generatin hidden outputs
        self.outputH = np.dot(self.w_IH, self.inputs) + self.bias_H
        #Activation function
        self.outputHA = activate(self.outputH)
        #Generating output
        self.output = np.dot(self.w_HO, self.outputHA) + self.bias_O
        #return matrix(activate(self.output), self.output.shape[0], self.output.shape[1])
        return np.array(activate(self.output))
    
    def train(self, input_array, target):
        #Cheking output without training
        salida = self.predict(input_array)
        #Convert target's list to matrix
        self.target = matrix(target, len(target), 1)
        #Calculating error
        self.error = self.target - salida
        #calculate Gradient = salida * ( 1 - salida)
        self.grad = np.array(dactivate(salida))
        self.grad *= self.error
        self.grad *= self.learning_rate
        #Calculate Deltas
        hiddenT = np.array(self.outputHA).T
        w_ho_delta = np.dot(self.grad, hiddenT)
        #Updating Weights and bias Hidden -> Outputs
        self.w_HO += w_ho_delta
        self.bias_O += self.grad
        #Calculating hidden layer error
        who_t = self.w_HO.T
        hidden_error = np.dot(who_t, self.error)
        #Calculating hidden gradient
        hidden_grad = np.array(dactivate(np.array(self.outputHA)))
        hidden_grad *= hidden_error
        hidden_grad *= self.learning_rate
        #Calculating input -> hidden delta
        inT = self.inputs.T
        wih_delta = np.dot(hidden_grad, inT)
        #Updating weights and bias inputs -> hidden
        self.w_IH += wih_delta
        self.bias_H += hidden_grad

    def set_learning_rate(learning_rate = 0.1):
        self.learning_rate = learning_rate


activate = lambda x: [[1 / (1 + exp(-x[r,c])) for c in range(x.shape[1])] for r in range(x.shape[0])]
dactivate = lambda x: [[x[r,c] * (1 - x[r,c]) for c in range(x.shape[1])] for r in range(x.shape[0])]
#matrix = lambda data, rows, cols: np.array([[data for i in range(cols)] for _ in range (rows)]) 

def matrix(data, rows, cols):
    if type(data) == list:
        return np.array([[data[_] for i in range(cols)] for _ in range (rows)])
    else:
        return np.array([[data() for i in range(cols)] for _ in range (rows)])
