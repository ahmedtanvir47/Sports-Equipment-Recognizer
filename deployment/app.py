from fastai.vision.all import *
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath



model = load_learner('models/sport-recognizer-v3.pkl')

equipment_labels = model.dls.vocab

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