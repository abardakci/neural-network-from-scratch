import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    pre_activation: np.array  

    @abstractmethod
    def forward(self, input_data: np.array) -> np.array:
        pass
    
    @abstractmethod
    def backpropagate(self, grad_output: np.array) -> np.array:
        pass

class ReLU(Activation):
    def forward(self, input_data: np.array) -> np.array:
        self.pre_activation = input_data
        activated_output = np.maximum(0, input_data)
        return activated_output
    
    def backpropagate(self, grad_output: np.array) -> np.array:
        relu_derivative = (self.pre_activation > 0).astype(float)
        return relu_derivative * grad_output

class Softmax(Activation):
    def forward(self, input_data: np.array) -> np.array:
        self.pre_activation = input_data
        exp_values = np.exp(input_data - np.max(input_data))  # Numerical stability
        activated_output = exp_values / np.sum(exp_values, axis=0)
        return activated_output
    
    def backpropagate(self, grad_output: np.array) -> np.array:
        return grad_output  # Softmax + Cross-Entropy special case
