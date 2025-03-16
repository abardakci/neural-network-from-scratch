import numpy as np
from activations import *

class Layer: # f( W.x + b ), f: activation function
    n_in: int 
    n_out: int # equal to neuron number of the layer
    W: np.array # W[n_out][n_in]
    bias: np.array # b[n_out]
    activation: Activation
    z: np.array
    
    def __init__(self, n_in: int, n_out: int, activation: Activation = ReLU):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation()
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
        self.bias = np.zeros((n_out, 1))

    def forward(self, z: np.array) -> np.array:
        yhat = np.dot(self.W, z) + self.bias # Wx + b
        yhat = self.activation.forward(yhat)  # Aktivasyon fonksiyonunu uygula
        self.z = z

        return yhat
        
    def backpropagate(self, dL_dx, learning_rate): # calculates gradien that should be backpropagated
        dL_dz = self.activation.backpropagate(dL_dx) # gradyen of activation function
        
        dW = np.dot(dL_dz, np.transpose(self.z)) # gradyen of W 
        db = dL_dz # gradyen of bias

        self.W -= learning_rate * dW
        self.bias -= learning_rate * db

        dL_dx = np.dot(np.transpose(self.W),  dL_dz) # backpropagate
        return dL_dx
