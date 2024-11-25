import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        # Placeholder for storing activations and gradients for visualization
        self.ac-tivations = {}
        self.gradients = {}

    def activate(self, z):
        """Apply activation function based on the chosen type."""
        if self.activation_fn == 'tanh':
            return np.tanh(z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function.")

    def activate_derivative(self, z):
        """Compute the derivative of the activation function."""
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_fn == 'relu':
            return (z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function.")

    def sigmoid(self, z):
        """Sigmoid activation function for the output layer."""
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Forward pass through the network."""
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = self.activate(self.z1)  # Apply activation function
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = self.sigmoid(self.z2)  # Sigmoid for binary classification

        # Store activations for visualization
        self.activations['input'] = X
        self.activations['hidden'] = self.a1
        self.activations['output'] = self.a2
        return self.a2

    def backward(self, X, y):
        """Backward pass to compute gradients and update weights."""
        m = X.shape[0]  # Number of training samples
        error_output = self.a2 - y  # Output error

        # Gradients for hidden-to-output weights and biases
        grad_weights_hidden_output = np.dot(self.a1.T, error_output) / m
        grad_bias_output = np.sum(error_output, axis=0, keepdims=True) / m

        # Error propagated to the hidden layer
        error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.activate_derivative(self.z1)

        # Gradients for input-to-hidden weights and biases
        grad_weights_input_hidden = np.dot(X.T, error_hidden) / m
        grad_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.weights_hidden_output -= self.lr * grad_weights_hidden_output
        self.bias_output -= self.lr * grad_bias_output
        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden

        # Store gradients for visualization
        self.gradients['weights_hidden_output'] = grad_weights_hidden_output
        self.gradients['bias_output'] = grad_bias_output
        self.gradients['weights_input_hidden'] = grad_weights_input_hidden
        self.gradients['bias_hidden'] = grad_bias_hidden


def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.activations['hidden']
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7,
    )
    ax_hidden.set_title("Hidden Space")

    # Input decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=np.linspace(0, 1, 10), cmap='bwr', alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space Decision Boundary")

    # Gradient visualization
    for i in range(mlp.weights_input_hidden.shape[0]):
        for j in range(mlp.weights_input_hidden.shape[1]):
            gradient = mlp.gradients['weights_input_hidden'][i, j]
            ax_gradient.arrow(
                i, j, gradient * 0.1, gradient * 0.1,
                color="blue" if gradient > 0 else "red",
                head_width=0.05,
                length_includes_head=True,
            )
    ax_gradient.set_title("Gradient Visualization")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)