import numpy as np 

np.set_printoptions(precision=2, suppress=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

ACTIVATION_FUNCTIONS = {
    "relu" : relu
}

ACTIVATION_FUNCTIONS_DERIVATIVES = {
    "relu" : relu_derivative
}

def mean_squared_error(expected_y, predicted_y):
    return np.mean((expected_y - predicted_y) ** 2)

class SimpNeuralNetwork:
    def __init__(self, name : str) -> None:
        self.name = name
        self.layers = []
        self.idx = 0

    def add_layer(self, n_neurons, activation_function=None, name="", layer_type="hidden"):
        if layer_type == "input":
            self.layers.append({
                "id" : self.idx, 
                "name" : name if name else f"input_{str(self.idx)}",
                "layer_type" : layer_type,
                "size" : n_neurons,
                "activation_function" : None
            })
            self.idx += 1

        if layer_type == "hidden":
            last_layer_size = self.layers[self.idx - 1]["size"]
            weights = np.random.random((last_layer_size, n_neurons))
            bias = np.random.random((1, n_neurons))
            self.layers.append({
                "id" : self.idx, 
                "name" : name if name else f"hidden_{str(self.idx)}",
                "layer_type" : layer_type,
                "size" : n_neurons,
                "weights" : weights,
                "bias" : bias,
                "activation_function" : activation_function
            })
            self.idx += 1

        if layer_type == "output":
            last_layer_size = self.layers[self.idx - 1]["size"]
            weights = np.random.random((last_layer_size, n_neurons))
            bias = np.random.random((1, n_neurons))
            self.layers.append({
                "id" : self.idx, 
                "name" : name if name else f"output_{str(self.idx)}",
                "layer_type" : layer_type,
                "size" : n_neurons,
                "weights" : weights,
                "bias" : bias,
                "activation_function" : None
            })
            self.idx += 1

    def add_multiple_hidden_layers(self, layers):
        for layer in layers:
            self.add_layer(n_neurons=layer[0], activation_function=layer[1])

    def remove_layer(self):
        pass 

    def _forward_layer(self, x, layer_id):
        y = np.dot(x, self.layers[layer_id]["weights"]) + self.layers[layer_id]["bias"]
        if self.layers[layer_id]["activation_function"] is None:
            return y
        return ACTIVATION_FUNCTIONS[self.layers[layer_id]["activation_function"]](y)
    
    def feed_forward(self, x):
        layers_output = []
        for layer_id in range(1, self.idx):
            layers_output.append(self._forward_layer(x if layer_id == 1 else layers_output[-1], layer_id))
        return layers_output
    
    def calculate_gradients(self, x, layers_output, expected_y):
        weights_gradients = []
        biases_gradients = []
        gradient_sum = None
        for layer_id in range(self.idx - 1, 0, -1):
            if layer_id == self.idx - 1:
                gradient_sum = layers_output[-1] - expected_y
            else:
                gradient_sum = np.dot(gradient_sum, self.layers[layer_id + 1]["weights"].T) * ACTIVATION_FUNCTIONS_DERIVATIVES[self.layers[layer_id]["activation_function"]](layers_output[layer_id - 1])
            if layer_id == 1:
                gradient_weights = np.dot(x.T, gradient_sum) 
                gradient_bias = np.sum(gradient_sum, axis=0, keepdims=True)
            else:
                gradient_weights = np.dot(layers_output[layer_id - 2].T, gradient_sum)
                gradient_bias = np.sum(gradient_sum, axis=0, keepdims=True)
            
            weights_gradients.insert(0, gradient_weights)
            biases_gradients.insert(0, gradient_bias)
        return weights_gradients, biases_gradients
    

    def update_weights_and_bias(self, weights_gradients, biases_gradients, learning_rate):
        for layer_id in range(1, self.idx):
            self.layers[layer_id]["weights"] -= learning_rate * weights_gradients[layer_id - 1]
            self.layers[layer_id]["bias"] -= learning_rate * biases_gradients[layer_id -1]

    def summary(self):
        print(f"Name : {self.name}")
        print(f"Layer   Type    Neurons    Activation")
        for l in self.layers:
            print(f"  {l['id']}    {l['layer_type']}       {l['size']}      {l['activation_function']}")

    def train(self, epochs, datasetX, datasetY, learning_rate):
        for epoch in range(epochs):
            for i in range(len(datasetX)):
                x =  np.array([datasetX[i]])
                y =  np.array([datasetY[i]])
                layers_output = self.feed_forward(x)
                weights_gradients, bias_gradients= self.calculate_gradients(x, layers_output, y)
                self.update_weights_and_bias(weights_gradients, bias_gradients, learning_rate)
