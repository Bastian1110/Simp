from FeedForward.SimpNeuralNetwork import SimpNeuralNetwork
import numpy as np
from FeedForward.display import plot_neural_network

nn = SimpNeuralNetwork("XOR")
nn.add_layer(2, layer_type="input")
nn.add_multiple_hidden_layers(
    [(3,"relu")]
)
nn.add_layer(1,activation_function="sigmoid", layer_type="output")
nn.summary()


dataset_x = np.array([[1, 0],[0,1], [0,0], [1,1]])
dataset_y = np.array([[1], [1], [0], [0]])

nn.train(50_000_000, dataset_x, dataset_y, 0.0001, batch_size=4)


for data in range(len(dataset_x)):
    print(f"Input : {dataset_x[data]} Output : {nn.feed_forward(dataset_x[data])[-1]}")

#plot_neural_network(nn.layers)