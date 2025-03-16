import numpy as np

from activations import *
from layer import Layer
from nn import NeuralNetwork

def main():
    train_path = "mnist_train.csv"
    test_path = "mnist_test.csv"

    train_data = np.loadtxt(train_path, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)

    y_train, x_train = train_data[:, 0].astype('int'), train_data[:, 1:]
    y_test, x_test = test_data[:, 0].astype('int'), test_data[:, 1:]
    
    num_classes = 10

    x_train = x_train.reshape(60000, -1, 1).astype('float32') / 255
    x_test = x_test.reshape(10000, -1, 1).astype('float32') / 255

    y_train = np.eye(num_classes)[y_train] # one-hot encode
    y_train = y_train.reshape(60000, -1, 1)

    y_test = np.eye(num_classes)[y_test] # one-hot encode
    y_test = y_test.reshape(10000, -1, 1)

    L1 = Layer(28*28, 64, ReLU) # 64 neuron for DenseLayer 1 with ReLU
    L2 = Layer(L1.n_out, 32, ReLU) # 32 neuron for DenseLayer 2 with ReLU
    L3 = Layer(L2.n_out, num_classes, Softmax) # Softmax output layer
    L = np.array([L1, L2, L3])
    nn = NeuralNetwork(L)
    
    TRAIN_SIZE = len(x_train)
    EPOCH_NUM = 1
    
    lr = 0.001    
    decay_rate = 0.95
    
    # Train loop
    for epoch in range(EPOCH_NUM):
        for i in range(TRAIN_SIZE):
            lr = lr * (decay_rate ** epoch)

            if i % (TRAIN_SIZE / 10) == 0:
                print("%", int(i / 60000 *100))

            yhat = nn.forward(x_train[i])
            
            nn.backpropagation(yhat, y_train[i], learning_rate=lr)

    TEST_SIZE = len(x_test)
    
    correct = 0
    for i in range(TEST_SIZE):
        x = x_test[i]
        y = y_test[i]

        yhat = nn.forward(x)

        predicted = np.argmax(yhat)
        label     = np.argmax(y)
        
        # print("label: ", label)
        # print("pred : ", predicted)
        
        if label == predicted:
            correct += 1
        
    accuracy = correct / TEST_SIZE
    print(f"Epoch {epoch + 1}, Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
   