from fastai.vision.all import *
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath



model = load_learner('models/sport-recognizer-v3.pkl')

equipment_labels = (
    'Archery Bow',
    'Badminton Shuttlecock',
    'Baseball Bat',
    'Basketball ball',
    'Bowling Ball',
    'Boxing Gloves',
    'Carrom board',
    'Chessboard',
    'Cricket Bat',
    'Frisbee disc',
    'Golf ball',
    'Hockey Stick',
    'Ice Skates',
    'Rugby Ball',
    'Skateboard',
    'Ski Poles',
    'Soccer ball',
    'Table Tennis Paddle',
    'Tennis Racket',
    'Volleyball ball'

)

def recognize_image(image):
  pred, idx, probs = model.predict(image)
  return dict(zip(equipment_labels, map(float, probs)))



#!export
image = gr.Image()
label = gr.Label()
examples = [
    'test_images/unknown_00.jpg',
    'test_images/unknown_01.jpg',
    'test_images/unknown_02.jpg',
    'test_images/unknown_03.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)