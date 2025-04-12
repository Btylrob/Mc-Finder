import tensorflow as tensorflow
import os
import cv2 
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt 

# GPU / CPU Mem config - I have a CPU however if you are running on GPU on comment out next line
#CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=""

# Image cleanup / remove all images from data with incorrect file exten / corrupted files / and < 10kb
data_dir= 'data'
image_exts = ['jpg', 'jpeg', 'png']

for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)

# Skip image if not found in directory
    if not os.path.isdir(class_path):
        continue
    
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            # check if file size is over 10kb
            if os.path.getsize(image_path) < 10 * 1024:
                print(f"Image is less than 10kb {image_path}")
                os.remove(image_path)
                continue

    