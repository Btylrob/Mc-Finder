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

            # Try to open images with PIL to check if corrupt
            with Image.open(image_path) as img: 
                img.verify()
            
            # Confirm valid extension
            ext = image_path.split('.')[-1].lower()
            if ext not in image_exts: 
                print(f"Image not a jpg, jpeg or png {image_path}")
                os.remove(image_path)

        # If any other files block code from executing 
        except Exception as e:
            print(f"Corrupted or unreadable image file {e}")
            os.remove(image_path)

    
# Load images from data directory 
data = tf.keras.util.image_dataset_from_directory(
    'data',
    image_size=(224, 224), # 224 pixels to imporove efficiency
    batch_size=32,
    shuffle=True
)

# get class names from data dir.
class_names = data.class_names

# Normalize images
data = data.map(lambda x, y: (x / 255.0, y))

# CNN Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)), #Input layer formats to 224px and rgb
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # applies 32 different filters 3x3 filters 
    tf.keras.layers.MaxPooling2D(), # picking the most prominent distinct features from our previous layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), # Flattens pooled map into a single vector for processing
    tf.keras.layers.Dense(128, activation='relu'), # makes accurate decision based on previous findings
    tf.keras.layers.Dense(len(class_names), activation='softmax') # turns vectors into a probability distroution 
])



    


            

    