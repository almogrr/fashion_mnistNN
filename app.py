import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.datasets import fashion_mnist
from enum import Enum

# Load MNIST Fashion dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the images
x_train = x_train.reshape(x_train.shape[0], 28*28) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28) / 255.0

# Number of classes, including "Not a Clothing Item"
num_classes = 11

# Convert labels to one-hot encoding and expand to 11 classes
y_train_one_hot = np.eye(10)[y_train]  # Original 10 classes
y_test_one_hot = np.eye(10)[y_test]

# Add a column for "Not a Clothing Item" class
y_train_one_hot = np.hstack((y_train_one_hot, np.zeros((y_train_one_hot.shape[0], 1))))
y_test_one_hot = np.hstack((y_test_one_hot, np.zeros((y_test_one_hot.shape[0], 1))))

# Enum for clothing labels
class ClothingLabel(Enum):
    T_SHIRT = 0
    TROUSER = 1
    PULLOVER = 2
    DRESS = 3
    COAT = 4
    SANDAL = 5
    SHIRT = 6
    SNEAKER = 7
    BAG = 8
    ANKLE_BOOT = 9
    NOT_CLOTHING = 10  # "Not a Clothing Item"

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = np.random.uniform(size=(input_size, hidden_size)) - 0.5
        self.hidden_bias = np.random.uniform(size=(1, hidden_size)) - 0.5
        self.output_weights = np.random.uniform(size=(hidden_size, output_size)) - 0.5
        self.output_bias = np.random.uniform(size=(1, output_size)) - 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        hidden_layer_activation = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = self.relu(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights)
        output_layer_activation += self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)

        return predicted_output

    def predict(self, inputs):
        prediction = self.forward(inputs)
        return np.argmax(prediction, axis=1)

    def train(self, x_train, y_train, x_val=None, y_val=None, learning_rate=0.1, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                # Get batch
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # Forward pass
                hidden_layer_activation = self.relu(np.dot(batch_x, self.hidden_weights) + self.hidden_bias)
                predicted_output = self.sigmoid(np.dot(hidden_layer_activation, self.output_weights) + self.output_bias)

                # Calculate error
                error = predicted_output - batch_y
                d_predicted_output = error * (predicted_output * (1 - predicted_output))

                # Calculate error_hidden_layer
                error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
                d_hidden_layer = error_hidden_layer * (hidden_layer_activation > 0)

                # Update weights and biases
                self.output_weights -= np.dot(hidden_layer_activation.T, d_predicted_output) * learning_rate
                self.output_bias -= np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
                self.hidden_weights -= np.dot(batch_x.T, d_hidden_layer) * learning_rate
                self.hidden_bias -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

            print(f"Epoch {epoch+1}/{epochs} completed")

            # Print validation accuracy
            if x_val is not None and y_val is not None:
                val_predictions = self.predict(x_val)
                val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
                print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy*100:.2f}%")

        # Save weights after training
        np.savez('model_weights.npz', hidden_weights=self.hidden_weights, hidden_bias=self.hidden_bias,
                output_weights=self.output_weights, output_bias=self.output_bias)
        print("Model trained and weights saved.")

    def test_random_image(self, x_data, y_data):
        # Select a random index
        idx = np.random.randint(len(x_data))
        image = x_data[idx].reshape(28, 28)  # Reshape for display
        label = y_data[idx]
        label = np.argmax(label) if len(label.shape) > 1 else label
        return image, label

# Instantiate the neural network
nn = NeuralNetwork(28*28, 128, num_classes)  # Output size now matches the number of classes

# GUI Application
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clothing Identifier")
        self.root.geometry("300x500")

        # Upload button
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Train button
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # Test button
        self.test_button = tk.Button(root, text="Test Random Image", command=self.test_random_image)
        self.test_button.pack(pady=10)

        # Label to display the image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Label to display the result
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

        # Load weights if available
        self.load_weights()

    def load_weights(self):
        try:
            data = np.load('model_weights.npz')
            nn.hidden_weights = data['hidden_weights']
            nn.hidden_bias = data['hidden_bias']
            nn.output_weights = data['output_weights']
            nn.output_bias = data['output_bias']
            print("Weights loaded successfully.")
        except FileNotFoundError:
            print("Weights not found. Please train the model first.")

    def train_model(self):
        nn.train(x_train[:, :28*28], y_train_one_hot[:, :10], x_val=x_test[:, :28*28], y_val=y_test_one_hot[:, :10], epochs=10)  # Adjust epochs as needed

    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open and convert the image to grayscale
            image = Image.open(file_path).convert('L')
            
            # Resize the image to 28x28 pixels
            image = image.resize((28, 28))
            
            # Convert the image to a numpy array and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Flatten the image array
            image_array = image_array.reshape(1, 28*28)
            
            # Display the image
            img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img)
            self.image_label.image = img
            
            # Predict the clothing item
            identified_label = nn.predict(image_array)
            label_text = ClothingLabel(identified_label[0]).name.replace('_', ' ').title()
            if label_text == "Not Clothing":
                label_text = "This is not a clothing item."
            self.result_label.config(text=f"Identified: {label_text}")



    def test_random_image(self):
        # Test random image from the dataset
        image, label = nn.test_random_image(x_test, y_test_one_hot)
        image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to 8-bit image for display
        img = ImageTk.PhotoImage(image_pil)

        # Display the image
        self.image_label.configure(image=img)
        self.image_label.image = img

        # Identify the clothing item
        identified_label = nn.predict(image.reshape(1, 28*28))
        label_text = ClothingLabel(identified_label[0]).name.replace('_', ' ').title()
        if label_text == "Not Clothing":
            label_text = "This is not a clothing item."
        self.result_label.config(text=f"Random Image Identified: {label_text}")

# Main loop
root = tk.Tk()
app = App(root)
root.mainloop()
