import numpy as np

class NN():
    def __init__(self, input_size, hidden_size, output_size, activation, dropout_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size])
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros([1, output_size])
        self.activation = activation
        self.placeholder = {"x": None, "y": None}
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()

    def forward(self, training=True):
        n = self.placeholder["x"].shape[0]
        self.a1 = self.placeholder["x"].dot(self.W1) + np.ones((n, 1)).dot(self.b1)
        self.h1 = np.maximum(self.a1, 0)  # ReLU Activation

        if training:
            self.dropout_mask = (np.random.rand(*self.h1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.h1 *= self.dropout_mask

        self.a2 = self.h1.dot(self.W2) + np.ones((n, 1)).dot(self.b2)

        if self.activation == "linear":
            self.y = self.a2.copy()
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.a2 - np.max(self.a2, 1, keepdims=True))
            self.y = self.y_logit / np.sum(self.y_logit, 1, keepdims=True)
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.a2))

        return self.y

    def backward(self):
        n = self.placeholder["x"].shape[0]

        self.grad_a2 = (self.y - self.placeholder["x"]) / n
        self.grad_b2 = np.sum(self.grad_a2, axis=0, keepdims=True)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)

        self.grad_h1 = self.grad_a2.dot(self.W2.T)
        self.grad_a1 = self.grad_h1 * (self.h1 > 0)

        if self.dropout_mask is not None:
            self.grad_a1 *= self.dropout_mask

        self.grad_b1 = np.sum(self.grad_a1, axis=0, keepdims=True)
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)

    def update(self, learning_rate=1e-3):
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2

    def computeLoss(self):
        if self.activation == "linear":
            return 0.5 * np.square(self.y - self.placeholder["x"]).mean()
        elif self.activation == "sigmoid":
            return -np.sum(self.placeholder["x"] * np.log(self.y + 1e-6) +
                           (1 - self.placeholder["x"]) * np.log(1 - self.y + 1e-6), axis=1).mean()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def get_layer_weights(self, layer_index):
        if layer_index == 0:
            return self.W1
        elif layer_index == 1:
            return self.W2
        else:
            raise ValueError("Invalid layer index. Valid indices are 0 or 1.")
