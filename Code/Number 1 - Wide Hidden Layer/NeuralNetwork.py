import numpy as np


class NN(): 
    def __init__(self, input_size, hidden_size, output_size, activation): 

        ##WEIGHT MATRIX  
        ##input_size is the vertical component and hidden size is the horizontal component
        ##np.random.randn(x,y) will create a matrix of size Y x X
        ##In this case Y will be the input size, while x is the hidden_size

        ##BIAS MATRIX
        ##Each node has its own bias that will be added to it

        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size])
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(input_size)
        self.b2 = np.zeros([1, output_size])
        # Activation Function & Placeholders
        self.activation = activation # “linear”/“softmax”/”sigmoid”
        self.placeholder = {"x":None, "y":None}
    
    # Feed Placeholder
    def feed(self, feed_dict):
        for key in feed_dict:
            ##maps the x and y values appropriate for what we are dealing with
            self.placeholder[key] = feed_dict[key].copy() 
    
    # Forward Propagation
    def forward(self):
        n = self.placeholder["x"].shape[0]

        ##Matrix multiplication of the input and weights, and then added with the biases to get a value
        self.a1 = self.placeholder["x"].dot(self.W1) + np.ones((n,1)).dot(self.b1)
        ##That value will then go through the activation function to determine what is activated
        ## h1 will be returned as a matrix, where each row corresponds to a sample and each column to a neuron
        self.h1 = np.maximum(self.a1,0) # ReLU Activation
        ## Afterwards we perform matrix multiplication before adding it with the bias before
        ##passing it through the appropriate activation function
        self.a2 = self.h1.dot(self.W2) + np.ones((n,1)).dot(self.b2)

        # Linear Activation
        if self.activation == "linear":
            self.y = self.a2.copy()
        # Softmax Activation
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.a2 - np.max(self.a2, 1, keepdims=True))
            self.y = self.y_logit / np.sum(self.y_logit, 1, keepdims=True)
        # Sigmoid Activation
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.a2))


        return self.y
    
    # Backward Propagation
    def backward(self):

        ##self.y is predicted output
        ##self.placeholder[y] is the actual, labeled output
        ##potential that y is in one hot encoding

        n = self.placeholder["y"].shape[0]


        #grad represents gradient, and our aim to minimize the gradient 

        #How loss change with respect to a2
        self.grad_a2 = (self.y - self.placeholder["y"]) / n
        self.grad_b2 = np.ones((n, 1)).T.dot(self.grad_a2)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)

        self.grad_h1 = self.grad_a2.dot(self.W2.T)

        #how loss change with respect to a2
        self.grad_a1 = self.grad_h1 * (self.h1 > 0)  # Assuming ReLU activation
        self.grad_b1 = np.ones((n, 1)).T.dot(self.grad_a1)
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)

    
    # Update Weights
    def update(self, learning_rate=1e-3):

        #Updates itself for backward propagation
        #Based on the gradient values, we multiplying with the learning rate and use the
        #result to know which step to take next
        self.W1 = self.W1 - learning_rate * self.grad_W1
        self.b1 = self.b1 - learning_rate * self.grad_b1
        self.W2 = self.W2 - learning_rate * self.grad_W2
        self.b2 = self.b2 - learning_rate * self.grad_b2
    
    # Loss Functions
    def computeLoss(self):
        loss = 0.0
        # Mean Square Error
        if self.activation == "linear":
            loss = 0.5 * np.square(self.y - self.placeholder["y"]).mean()
        # Softmax Cross Entropy
        elif self.activation == "softmax":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6)
            loss = np.sum(loss, 1).mean()
        # Sigmoid Cross Entropy
        elif self.activation == "sigmoid":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6) - \
            (1-self.placeholder["y"]) * np.log(1-self.y + 1e-6)
            loss = np.mean(loss)
        return loss