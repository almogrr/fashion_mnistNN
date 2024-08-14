import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.datasets import fashion_mnist
from enum import Enum

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

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = np.random.uniform(size=(input_size, hidden_size)) - 0.5
        self.hidden_bias = np.random.uniform(size=(hidden_size,)) - 0.5
        self.output_weights = np.random.uniform(size=(hidden_size, output_size)) - 0.5
        self.output_bias = np.random.uniform(size=(output_size,)) - 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer_activation = np.dot(inputs, self.hidden_weights)
        self.hidden_layer_activation += self.hidden_bias
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.output_weights)
        self.output_layer_activation += self.output_bias
        predicted_output = self.sigmoid(self.output_layer_activation)

        return predicted_output

    def backward(self, inputs, target, predicted_output, learning_rate):
        error = target - predicted_output
        d_predicted_output = error * self.sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)

        self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0) * learning_rate
        self.hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0) * learning_rate

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(x_train.shape[0]):
                inputs = x_train[i:i+1]
                target = np.zeros(self.output_size)
                target[y_train[i]] = 1
                predicted_output = self.forward(inputs)
                self.backward(inputs, target, predicted_output, learning_rate)
            if epoch % 10 == 0:
                loss = np.mean(np.square(target - self.forward(x_train)))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, inputs):
        prediction = self.forward(inputs)
        return np.argmax(prediction)

# Prepare and normalize data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0

# Instantiate and train the neural network
nn = NeuralNetwork(28*28, 128, 10)
nn.train(x_train, y_train, epochs=50, learning_rate=0.1)

# Calculate success rate
def calculate_success_rate():
    x_test = x_test.reshape(-1, 28*28) / 255.0
    correct_predictions = 0
    for i in range(x_test.shape[0]):
        inputs = x_test[i:i+1]
        prediction = nn.predict(inputs)
        if prediction == y_test[i]:
            correct_predictions += 1
    success_rate = correct_predictions / x_test.shape[0] * 100
    print(f"Success Rate: {success_rate:.2f}%")

calculate_success_rate()

# GUI Application
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clothing Identifier")
        self.root.geometry("300x400")

        # Upload button
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        # Label to display the image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Label to display the result
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path).resize((28, 28)).convert('L')
            image_array = np.array(image).reshape(1, 28*28)
            image_array = image_array / 255.0  # Normalize the image

            # Display the image
            img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img)
            self.image_label.image = img

            # Identify the clothing item
            identified_label = nn.predict(image_array)
            self.result_label.config(text=f"Identified: {ClothingLabel(identified_label).name.replace('_', ' ').title()}")

# Main loop
root = tk.Tk()
app = App(root)
root.mainloop()
