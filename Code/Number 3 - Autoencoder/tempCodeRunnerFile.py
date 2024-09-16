import numpy as np
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def OneHot(y):
    y_one_hot = np.eye(10, dtype=np.float32)[y]
    return y_one_hot

def Accuracy(y,y_):
    y_digit = np.argmax(y,1)
    y_digit_ = np.argmax(y_,1)
    temp = np.equal(y_digit, y_digit_).astype(np.float32)
    return np.sum(temp) / float(y_digit.shape[0])

if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, _ = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, _ = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Data Processing
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    # Create NN Model
    nn = NeuralNetwork.NN(784,128,784,10,"sigmoid")

    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        
        # Sample Data Batch
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]

        # Forward & Backward & Update
        nn.feed({"x": x_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-2)

        # Compute Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)

        if i % 100 == 0:
            print("\r[Iteration {:5d}] Loss={:.4f}".format(i, loss))

    # Display a few reconstructed images
    nn.feed({"x": x_test})
    x_test_reconstructed = nn.forward()

    n = 5  # Number of images to display
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(2, n, i + n + 1)
        plt.imshow(x_test_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
    plt.show()

    plt.plot(loss_rec)
    plt.title('Training Loss')
    plt.show()