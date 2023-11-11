
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

# use the BiggerNet model from pale_nets.py
# and load the weights from the file bigger_net.pth
# which was trained in mnist_part_4.ipynb
import models.pale_nets as pale_nets
model = pale_nets.BiggerNet()
model.load_state_dict(torch.load('models/bigger_net_mnist.pth'))
model.eval()

window = tk.Tk()
window.title("Draw a digit")

canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

image = Image.new('L', (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(image)

def predict_digit():
    # turn the image into a normalised tensor of the right size [1, 28, 28]
    resized_image = image.resize((28, 28))
    np_image = np.array(resized_image)
    np_image = np_image / 255.0
    np_image = np_image[np.newaxis, :, :]
    tensor_image = torch.from_numpy(np_image).float()

    with torch.no_grad():
        output = model(tensor_image)
        predicted_digit = torch.argmax(output).item()

    prediction_label.config(text=f"Predicted digit: {predicted_digit}")

def handle_mouse(event):
    x, y = event.x, event.y
    radius = 10
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill='black')
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='white')

def erase_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill='black')

predict_button = tk.Button(window, text="Predict digit", command=predict_digit)
predict_button.pack()

erase_button = tk.Button(window, text="Erase", command=erase_canvas)
erase_button.pack()

prediction_label = tk.Label(window, text="")
prediction_label.pack()

canvas.bind("<B1-Motion>", handle_mouse)

window.mainloop()
