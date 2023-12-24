import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Loading the saved model
loaded_model = load_model("C:\\Users\\tejus\\Desktop\\6th Sem Project\\Malaria Detection using CNN\\Version 1\\bestmodel.h5")

def preprocess_image(file_path):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(75, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

    # Check if a file is selected
    if file_path:
        # Load and preprocess the selected image
        image_array = preprocess_image(file_path)

        # Perform malaria detection using the loaded model
        prediction = loaded_model.predict(image_array)

        # Display the result in the output label
        if prediction[0, 0] > 0.5:
            result_label.config(text="Test results indicate a positive presence of malaria.", fg="red")
        else:
            result_label.config(text="Test results indicate the absence of malaria.", fg="green")

        # Display the selected image in the GUI
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create the main application window
app = tk.Tk()
app.title("Malaria Detection App")
app.geometry("640x480")  # Set fixed size

# Create and place the "Select Image" button
select_button = tk.Button(app, text="Select Image", command=classify_image, width=20, height=2)
select_button.pack(pady=10)

# Create and place the output label
result_label = tk.Label(app, text="", font=("Helvetica", 16), pady=20)
result_label.pack()

# Create and place the label to display the selected image
image_label = tk.Label(app)
image_label.pack()

# Start the application
app.mainloop()
