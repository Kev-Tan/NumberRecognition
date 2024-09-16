import numpy as np
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def OneHot(y):
    y_one_hot = np.eye(10, dtype=np.float32)[y]
    return y_one_hot

def Accuracy(y, y_):
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
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
    nn = NeuralNetwork.NN(784, 128, 784, "sigmoid", dropout_rate=0.5)

    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        
        # Sample Data Batch
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]
        
        # Add noise for denoising autoencoder
        noise_factor = 0.5
        x_batch_noisy = x_batch + noise_factor * np.random.randn(*x_batch.shape)
        x_batch_noisy = np.clip(x_batch_noisy, 0., 1.)

        # Forward & Backward & Update
        nn.feed({"x": x_batch_noisy})
        nn.forward(training=True)
        nn.backward()
        nn.update(1e-2)

        # Compute Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)

        if i % 100 == 0:
            print("\r[Iteration {:5d}] Loss={:.4f}".format(i, loss))

    # Display a few reconstructed images and noisy inputs
    nn.feed({"x": x_test})
    x_test_reconstructed = nn.forward(training=False)

    n = 5  # Number of images to display
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Add noisy input
        noise_factor = 0.5
        x_test_noisy = x_test[i] + noise_factor * np.random.randn(*x_test[i].shape)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        plt.subplot(3, n, i + n + 1)
        plt.imshow(x_test_noisy.reshape(28, 28), cmap='gray')
        plt.title("Noisy Input")
        plt.axis('off')

        plt.subplot(3, n, i + 2 * n + 1)
        plt.imshow(x_test_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.plot(loss_rec)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    # Visualize the first 16 filters
    weights = nn.get_layer_weights(0)  # Get weights of the first layer
    filters = weights.T[:16]  # Taking the first 16 filters

    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(filters[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"Filter {i+1}")
    plt.show()
