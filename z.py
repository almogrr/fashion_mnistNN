import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.datasets import fashion_mnist
from enum import Enum

# Load MNIST Fashion dataset
(_, _), (x_test, y_test) = fashion_mnist.load_data()

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
        self.hidden_bias = np.random.uniform(size=(1, hidden_size)) - 0.5
        self.output_weights = np.random.uniform(size=(hidden_size, output_size)) - 0.5
        self.output_bias = np.random.uniform(size=(1, output_size)) - 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer_activation = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights)
        output_layer_activation += self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)

        return predicted_output

    def predict(self, inputs):
        prediction = self.forward(inputs)
        return np.argmax(prediction)

# Instantiate the neural network
nn = NeuralNetwork(28*28, 128, 10)

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
