import numpy as np
import tensorflow as tf

# Step 1: Fetch the Fashion MNIST Data
class FashionMNISTLoader:
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        # Normalize the images
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        # Flatten the images for the neural network
        self.train_images = self.train_images.reshape(self.train_images.shape[0], -1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], -1)

# Step 2: Define the Neural Network Class
class FashionMNISTModel:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2
    
    def compute_loss(self, A2, Y):
        m = Y.shape[0]
        logprobs = -np.log(A2[range(m), Y])
        loss = np.sum(logprobs) / m
        return loss
    
    def backward_propagation(self, X, Y, Z1, A1, A2):
        m = X.shape[0]
        dZ2 = A2
        dZ2[range(m), Y] -= 1
        dZ2 /= m

        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2, learning_rate=0.01):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, Y, iterations=1000, learning_rate=0.01):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X)
            loss = self.compute_loss(A2, Y)
            dW1, db1, dW2, db2 = self.backward_propagation(X, Y, Z1, A1, A2)
            self.update_parameters(dW1, db1, dW2, db2, learning_rate)
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss}")
    
    def predict(self, X):
        _, _, _, A2 = self.forward_propagation(X)
        predictions = np.argmax(A2, axis=1)
        return predictions

# Step 3: Use the Model with Examples
if __name__ == "__main__":
    # Load data
    loader = FashionMNISTLoader()
    train_images = loader.train_images
    train_labels = loader.train_labels
    test_images = loader.test_images
    test_labels = loader.test_labels

    # Instantiate the model and train
    model = FashionMNISTModel()
    model.train(train_images, train_labels)

    # Examples to test the model
    test_examples = test_images[:5]
    test_predictions = model.predict(test_examples)

    # Print predictions
    for i, prediction in enumerate(test_predictions):
        print(f"Test Example {i+1}: Predicted class = {prediction}, Actual class = {test_labels[i]}")

    # Expected output examples:
    # Example 1: Pants (class 1)
    # Example 2: Shirt (class 6)
    # Example 3: Sneaker (class 7)
    # Example 4: Dress (class 3)
    # Example 5: Coat (class 4)
