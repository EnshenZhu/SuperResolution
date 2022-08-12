import cv2
from cv2 import dnn_superres
import os

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
file_name = 'test1'
image = cv2.imread('./input_img/' + file_name + '.jpg')

# Read the desired model
path = "./sr_model/LapSRN_x8.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("lapsrn", 8)

# Upscale the image
result = sr.upsample(image)

# Save the image
if not cv2.imwrite(os.path.join("./result_img/upscaled_" + file_name + ".png"), result):
    raise Exception("Could not write image")
