# Bi-Polar Step activation function
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

# Sigmoid activation function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# ReLU activation function
def relu_function(x):
    return max(0, x)

import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Function to train perceptron for AND gate logic
def train_and_perceptron(initial_weights, learning_rate):
    # Initial weights
    W = np.array(initial_weights)

    # Training data for AND gate
    inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    targets = np.array([0, 0, 0, 1])

    # Training the perceptron
    epochs = 0
    errors = []

    while True:
        error_sum = 0

        for i in range(len(inputs)):
            # Calculate output
            output = step_function(np.dot(inputs[i], W))

            # Update weights
            W += learning_rate * (targets[i] - output) * inputs[i]

            # Calculate error
            error_sum += (targets[i] - output) ** 2

        # Store error for plotting
        errors.append(error_sum)

        # Check convergence condition
        if error_sum <= 0.002 or epochs >= 1000:
            break

        epochs += 1

    return epochs, errors

# Function to train perceptron with different activation functions
def train_perceptron_with_activations(inputs, targets, activation_functions):
    results = []

    for activation_function in activation_functions:
        W = np.array([10, 0.2, -0.75])
        alpha = 0.05
        epochs = 0
        errors = []

        while True:
            error_sum = 0

            for i in range(len(inputs)):
                # Calculate output using the specified activation function
                output = activation_function(np.dot(inputs[i], W))

                # Update weights
                W += alpha * (targets[i] - output) * inputs[i]

                # Calculate error
                error_sum += (targets[i] - output) ** 2

            # Store error for plotting
            errors.append(error_sum)

            # Check convergence condition
            if error_sum <= 0.002 or epochs >= 1000:
                break

            epochs += 1

        results.append((epochs, errors))

    return results

# Function to train perceptron with different learning rates
def train_perceptron_with_learning_rates(learning_rates):
    epochs_list = []

    for rate in learning_rates:
        epochs = train_and_perceptron([10, 0.2, -0.75], rate)[0]
        epochs_list.append(epochs)

    return epochs_list

# Function to train perceptron for XOR gate logic
def train_xor_perceptron():
    # Initial weights
    W = np.array([10, 0.2, -0.75])

    # XOR gate inputs and targets
    xor_inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    xor_targets = np.array([0, 1, 1, 0])

    # Learning rate
    alpha = 0.1

    # Training the perceptron
    epochs = 0
    errors = []

    while True:
        error_sum = 0

        for i in range(len(xor_inputs)):
            # Calculate output
            output = step_function(np.dot(xor_inputs[i], W))

            # Update weights
            W += alpha * (xor_targets[i] - output) * xor_inputs[i]

            # Calculate error
            error_sum += (xor_targets[i] - output) ** 2

        # Store error for plotting
        errors.append(error_sum)

        # Check convergence condition
        if error_sum == 0 or epochs >= 1000:
            break

        epochs += 1

    return epochs, errors

# Training data 
inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
targets = np.array([0, 0, 0, 1])

# A1
and_epochs, and_errors = train_and_perceptron([10, 0.2, -0.75], 0.05)
plt.plot(range(and_epochs + 1), and_errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Convergence of Perceptron for AND Gate Logic')
plt.show()
print(f"Number of epochs needed for convergence (AND gate): {and_epochs}")

# A2
import matplotlib.pyplot as plt

# A2
activation_functions = [bipolar_step_function, sigmoid_function, relu_function]
results_activations = train_perceptron_with_activations(inputs, targets, activation_functions)

# Plotting
for i, (epochs, errors) in enumerate(results_activations):
    plt.plot(range(epochs + 1), errors, label=f'Activation Function {i + 1}')

plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Convergence of Perceptron with Different Activation Functions')
plt.legend()
plt.show()

# A3
learning_rates_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epochs_list_learning_rates = train_perceptron_with_learning_rates(learning_rates_to_test)

plt.plot(learning_rates_to_test, epochs_list_learning_rates, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Impact of Learning Rate on Convergence')
plt.show()

# A4
xor_epochs, xor_errors = train_xor_perceptron()
plt.plot(range(xor_epochs + 1), xor_errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Convergence of Perceptron for XOR Gate Logic')
plt.show()
print(f"Number of epochs needed for XOR gate logic: {xor_epochs}")
#A5,6
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and learning rate for perceptron learning
perceptron_weights = np.random.rand(6)  # Adjusted to match the number of features (including bias)
learning_rate = 0.01

# Customer data
data = [
    [1, 20, 6, 2, 386, 1],   # High Value
    [2, 16, 3, 6, 289, 1],   # High Value
    [3, 27, 6, 2, 393, 1],   # High Value
    [4, 19, 1, 2, 110, 0],   # Low Value
    [5, 24, 4, 2, 280, 1],   # High Value
    [6, 22, 1, 5, 167, 0],   # Low Value
    [7, 15, 4, 2, 271, 1],   # High Value
    [8, 18, 4, 2, 274, 1],   # High Value
    [9, 21, 1, 4, 148, 0],   # Low Value
    [10, 16, 2, 4, 198, 0],  # Low Value
]

# Separate features and labels
X = np.array([item[:-1] for item in data])
y = np.array([item[-1] for item in data])

# Train the perceptron
epochs = 1000
perceptron_errors = []

for epoch in range(epochs):
    perceptron_error_sum = 0

    for i in range(len(X)):
        # Add bias term to the features
        features_with_bias = np.concatenate(([1], X[i]))

        # Calculate weighted sum
        weighted_sum = np.dot(features_with_bias, perceptron_weights)

        # Apply sigmoid activation function
        predicted_output = sigmoid_function(weighted_sum)

        # Update weights
        perceptron_weights += learning_rate * (y[i] - predicted_output) * predicted_output * (1 - predicted_output) * features_with_bias

        # Calculate error
        perceptron_error_sum += 0.5 * (y[i] - predicted_output) ** 2

    perceptron_errors.append(perceptron_error_sum)

# Calculate weights using matrix pseudo-inverse
X_pseudo_inverse = np.linalg.pinv(np.concatenate((np.ones((len(X), 1)), X), axis=1))
weights_pseudo_inverse = np.dot(X_pseudo_inverse, y)

# Generate predictions using weights from matrix pseudo-inverse
predictions_pseudo_inverse = sigmoid_function(np.dot(np.concatenate((np.ones((len(X), 1)), X), axis=1), weights_pseudo_inverse))

# Calculate error using matrix pseudo-inverse
pseudo_inverse_error = 0.5 * np.sum((y - predictions_pseudo_inverse) ** 2)

# Print and compare errors
print(f"Perceptron Learning Error: {perceptron_errors[-1]}")
print(f"Matrix Pseudo-Inverse Error: {pseudo_inverse_error}")

# Plotting the error convergence for perceptron learning
plt.plot(range(epochs), perceptron_errors, label='Perceptron Learning')
plt.axhline(y=pseudo_inverse_error, color='r', linestyle='--', label='Matrix Pseudo-Inverse')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error Convergence Comparison')
plt.legend()
plt.show()

#A10
from sklearn.neural_network import MLPClassifier
import numpy as np

# AND Gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR Gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create MLPClassifier for AND Gate
mlp_and = MLPClassifier(learning_rate_init=0.01, momentum=0.9)

# Create MLPClassifier for XOR Gate
mlp_xor = MLPClassifier(learning_rate_init=0.01, momentum=0.9)

# Train MLPClassifier for AND Gate
mlp_and.fit(X_and, y_and)

# Train MLPClassifier for XOR Gate
mlp_xor.fit(X_xor, y_xor)

# Test the trained models
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Predictions for AND Gate
predictions_and = mlp_and.predict(test_data)

# Predictions for XOR Gate
predictions_xor = mlp_xor.predict(test_data)

# Display results
print("AND Gate Predictions:", predictions_and)
print("XOR Gate Predictions:", predictions_xor)
#A11
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data(excel_path):
    # Load data from Excel sheet
    df = pd.read_excel(excel_path)

    # Separate features and labels
    features = df.drop('Disease', axis=1).to_numpy()
    labels = df['Disease'].to_numpy()

    return features, labels

# Assuming your dataset is stored in a variable called 'your_dataset'
excel_path = "C:/Users/vishn/Downloads/extracted_features.xlsx"
X, y = load_data(excel_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLPClassifier
mlp_classifier = MLPClassifier(learning_rate_init=0.05, momentum=0.9)

# Fit the model to the training data
mlp_classifier.fit(X_train, y_train)

# Predict disease labels on the test set
predictions = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print(f"Accuracy: {accuracy}")
#A7
import numpy as np

# AND gate input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Neural Network parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.05
epochs = 1000
error_threshold = 0.002

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Calculate error
    error = y - final_output

    # Check for convergence
    if np.mean(np.abs(error)) <= error_threshold:
        print(f"Converged at epoch {epoch + 1}")
        break

    # Backward pass
    output_error = error * sigmoid_derivative(final_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(output_error) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

# Test the trained model
predicted_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)
print("Final Predicted Output:")
print(predicted_output)
#A8
import numpy as np

# XOR gate input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Neural Network parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.05
epochs = 10000
error_threshold = 0.002

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Calculate error
    error = y - final_output

    # Check for convergence
    if np.mean(np.abs(error)) <= error_threshold:
        print(f"Converged at epoch {epoch + 1}")
        break

    # Backward pass
    output_error = error * sigmoid_derivative(final_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(output_error) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

# Print the weights
print("Input Weights to Hidden Layer:")
print(weights_input_hidden)

print("\nHidden Weights to Output Layer:")
print(weights_hidden_output)

# Test the trained model
predicted_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)
print("\nFinal Predicted Output:")
print(predicted_output)
#A9a
import numpy as np

# AND gate input and output
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

# Neural Network parameters
input_size_and = 2
hidden_size_and = 2
output_size_and = 2
learning_rate_and = 0.05
epochs_and = 1000
error_threshold_and = 0.002

# Initialize weights and biases
weights_input_hidden_and = np.random.rand(input_size_and, hidden_size_and)
weights_hidden_output_and = np.random.rand(hidden_size_and, output_size_and)
bias_hidden_and = np.zeros((1, hidden_size_and))
bias_output_and = np.zeros((1, output_size_and))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network for AND gate
for epoch in range(epochs_and):
    # Forward pass
    hidden_input_and = np.dot(X_and, weights_input_hidden_and) + bias_hidden_and
    hidden_output_and = sigmoid(hidden_input_and)
    final_input_and = np.dot(hidden_output_and, weights_hidden_output_and) + bias_output_and
    final_output_and = sigmoid(final_input_and)

    # Calculate error
    error_and = y_and - final_output_and

    # Check for convergence
    if np.mean(np.abs(error_and)) <= error_threshold_and:
        print(f"Converged at epoch {epoch + 1}")
        break

    # Backward pass
    output_error_and = error_and * sigmoid_derivative(final_output_and)
    hidden_layer_error_and = output_error_and.dot(weights_hidden_output_and.T) * sigmoid_derivative(hidden_output_and)

    # Update weights and biases
    weights_hidden_output_and += hidden_output_and.T.dot(output_error_and) * learning_rate_and
    weights_input_hidden_and += X_and.T.dot(hidden_layer_error_and) * learning_rate_and
    bias_output_and += np.sum(output_error_and, axis=0, keepdims=True) * learning_rate_and
    bias_hidden_and += np.sum(hidden_layer_error_and, axis=0, keepdims=True) * learning_rate_and

# Print the weights for AND gate
print("AND Gate - Input Weights to Hidden Layer:")
print(weights_input_hidden_and)

print("\nAND Gate - Hidden Weights to Output Layer:")
print(weights_hidden_output_and)

# Test the trained model for AND gate
predicted_output_and = sigmoid(np.dot(sigmoid(np.dot(X_and, weights_input_hidden_and) + bias_hidden_and), weights_hidden_output_and) + bias_output_and)
print("\nAND Gate - Final Predicted Output:")
print(predicted_output_and)
#A9b
# XOR gate input and output
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Neural Network parameters
input_size_xor = 2
hidden_size_xor = 2
output_size_xor = 2
learning_rate_xor = 0.05
epochs_xor = 10000
error_threshold_xor = 0.002

# Initialize weights and biases
weights_input_hidden_xor = np.random.rand(input_size_xor, hidden_size_xor)
weights_hidden_output_xor = np.random.rand(hidden_size_xor, output_size_xor)
bias_hidden_xor = np.zeros((1, hidden_size_xor))
bias_output_xor = np.zeros((1, output_size_xor))

# Training the neural network for XOR gate
for epoch in range(epochs_xor):
    # Forward pass
    hidden_input_xor = np.dot(X_xor, weights_input_hidden_xor) + bias_hidden_xor
    hidden_output_xor = sigmoid(hidden_input_xor)
    final_input_xor = np.dot(hidden_output_xor, weights_hidden_output_xor) + bias_output_xor
    final_output_xor = sigmoid(final_input_xor)

    # Calculate error
    error_xor = y_xor - final_output_xor

    # Check for convergence
    if np.mean(np.abs(error_xor)) <= error_threshold_xor:
        print(f"Converged at epoch {epoch + 1}")
        break

    # Backward pass
    output_error_xor = error_xor * sigmoid_derivative(final_output_xor)
    hidden_layer_error_xor = output_error_xor.dot(weights_hidden_output_xor.T) * sigmoid_derivative(hidden_output_xor)

    # Update weights and biases
    weights_hidden_output_xor += hidden_output_xor.T.dot(output_error_xor) * learning_rate_xor
    weights_input_hidden_xor += X_xor.T.dot(hidden_layer_error_xor) * learning_rate_xor
    bias_output_xor += np.sum(output_error_xor, axis=0, keepdims=True) * learning_rate_xor
    bias_hidden_xor += np.sum(hidden_layer_error_xor, axis=0, keepdims=True) * learning_rate_xor

# Print the weights for XOR gate
print("\nXOR Gate - Input Weights to Hidden Layer:")
print(weights_input_hidden_xor)

print("\nXOR Gate - Hidden Weights to Output Layer:")
print(weights_hidden_output_xor)

# Test the trained model for XOR gate
predicted_output_xor = sigmoid(np.dot(sigmoid(np.dot(X_xor, weights_input_hidden_xor) + bias_hidden_xor), weights_hidden_output_xor) + bias_output_xor)
print("\nXOR Gate - Final Predicted Output:")
print(predicted_output_xor)