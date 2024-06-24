import numpy as np 

np.set_printoptions(precision=2, suppress=True)

class SimpNeuralNetwork:
    def __init__(self, name : str) -> None:
        self.name = name
        self.layers = []
        self.idx = 0

    def add_layer(self, n_neurons, activation_function="relu", name="", layer_type="hidden"):
        if layer_type == "input":
            self.layers.append({
                "id" : self.idx, 
                "name" : name if name else f"input_{str(self.idx)}",
                "layer_type" : layer_type,
                "size" : n_neurons
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
                "activation_function" : activation_function
            })
            self.idx += 1

    def remove_layer():
        pass 

    def summary(self):
        print(f"{self.name} : ")
        for l in self.layers:
            print(l)


