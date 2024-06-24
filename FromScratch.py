import numpy as np 
import random 

np.set_printoptions(precision=2, suppress=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def forward_layer(x, weights, bias, activation=None):
    z = np.dot(x, weights) + bias
    if activation is None:
        return z
    return activation(z)

def mean_squared_error(expected_y, predicted_y):
    return np.mean((expected_y - predicted_y) ** 2)

learning_rate = 0.01

weights_input_to_hidden_one = np.random.random((2, 3))
bias_hidden_one = np.random.random((1, 3))

weights_hidden_one_to_hidden_two = np.random.random((3, 3))
bias_hidden_two = np.random.random((1, 3))

weights_hidden_two_to_output = np.random.random((3, 1))
bias_output = np.random.random((1, 1))

datasetX = np.array([[1, 2], [4, 1], [0, 0], [2, 3]])
datasetY = np.array([[3], [5], [0], [5]])

def generate_batch(dataset_x, dataset_y, batch_size):
    numbers = list(range(len(dataset_x)))
    random.shuffle(numbers)
    batches = []
    for i in range(0, len(numbers), batch_size):
        batch_x = np.array([dataset_x[j] for j in numbers[i:i+batch_size]])
        batch_y = np.array([dataset_y[j] for j in numbers[i:i+batch_size]])
        batches.append((batch_x, batch_y))
    return batches

batch_size = 2
iterations = 10000
for i in range(iterations):
    batches = generate_batch(datasetX, datasetY, batch_size)
    for batch in batches:
        test_input = np.array(batch[0])
        test_output = np.array(batch[1])

        # Forward pass
        result_hidden_one = forward_layer(test_input, weights_input_to_hidden_one, bias_hidden_one, relu)
        result_hidden_two = forward_layer(result_hidden_one, weights_hidden_one_to_hidden_two, bias_hidden_two, relu)
        result_output = forward_layer(result_hidden_two, weights_hidden_two_to_output, bias_output)

        print(f"Inout : {test_input}")
        print(f"Output : {test_output}")

        # Calculate loss
        loss = mean_squared_error(test_output, result_output)

        # Backpropagation
        gradient_sum_output = result_output - test_output
        gradient_output_to_hidden_two = np.dot(result_hidden_two.T, gradient_sum_output) / batch_size
        gradient_bias_output = np.sum(gradient_sum_output, axis=0, keepdims=True) / batch_size

        gradient_sum_hidden_two = np.dot(gradient_sum_output, weights_hidden_two_to_output.T) * relu_derivative(result_hidden_two)
        gradient_hidden_one_to_hidden_two = np.dot(result_hidden_one.T, gradient_sum_hidden_two) / batch_size
        gradient_bias_hidden_two = np.sum(gradient_sum_hidden_two, axis=0, keepdims=True) / batch_size

        gradient_sum_hidden_one = np.dot(gradient_sum_hidden_two, weights_hidden_one_to_hidden_two.T) * relu_derivative(result_hidden_one)
        gradient_input_to_hidden_one = np.dot(test_input.T, gradient_sum_hidden_one) / batch_size
        gradient_bias_hidden_one = np.sum(gradient_sum_hidden_one, axis=0, keepdims=True) / batch_size

        # Update weights and biases
        weights_input_to_hidden_one -= learning_rate * gradient_input_to_hidden_one
        weights_hidden_one_to_hidden_two -= learning_rate * gradient_hidden_one_to_hidden_two
        weights_hidden_two_to_output -= learning_rate * gradient_output_to_hidden_two

        bias_hidden_one -= learning_rate * gradient_bias_hidden_one
        bias_hidden_two -= learning_rate * gradient_bias_hidden_two
        bias_output -= learning_rate * gradient_bias_output

    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss}")


