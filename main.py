from SimpNeuralNetwork import SimpNeuralNetwork
import numpy as np
from display import plot_neural_network

nn = SimpNeuralNetwork("XOR")
nn.add_layer(2, layer_type="input")
nn.add_multiple_hidden_layers(
    [(3,"relu"), 
     (3,"relu")]
)
nn.add_layer(1, activation_function="none", layer_type="output")
nn.summary()


dataset_x = np.array([[1, 0], [0,0], [1,1], [0,1]])
dataset_y = np.array([[1], [0], [0], [1]])

nn.train(10000, dataset_x, dataset_y, 0.01)


for data in range(len(dataset_x)):
    print(f"Input : {dataset_x[data]} Output : {nn.feed_forward(dataset_x[data])[-1]}")

plot_neural_network(nn.layers)