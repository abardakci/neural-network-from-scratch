import numpy as np

def bce_softmax_gradien(yhat: np.array, y: np.array):
    return yhat - y

class NeuralNetwork:
    layers: np.array

    def __init__(self, L: np.array):
        self.layers = L

    def forward(self, nn_input: np.array):
        z = nn_input
        for layer in self.layers:
            z = layer.forward(z)
            
        return z # yhat (10,)
    
    def backpropagation(self, yhat: np.array, y: np.array, learning_rate=0.0001):
        dL_dx = bce_softmax_gradien(yhat, y)
        i = len(self.layers) - 1
        while i >= 0:
            dL_dx = self.layers[i].backpropagate(dL_dx, learning_rate)
            i -= 1
