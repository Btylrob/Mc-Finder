import tensorflow as tf 
import os
import cv2 
from PIL import Image
import numpy as np
import gradio as gr 
from matplotlib import pyplot as plt 

# GPU / CPU Mem config - I have a CPU however if you are running on GPU on comment out next line
#CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
data = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split = .2,
    subset = 'training',
    seed = 123,
    image_size=(224, 224), 
    batch_size=32,
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split=0.2,     
    subset='validation',      # This is the validation portion
    seed=123, # controlls randomess
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# get class names from data dir.
class_names = data.class_names

# Normalize images
data = data.map(lambda x, y: (x / 255.0, y))
val_data = val_data.map(lambda x, y: (x / 255.0, y))


# randomizes display of data flipping it and giving it versatility
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# CNN Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)), #Input layer formats to 224px and rgb
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # applies 32 different filters 3x3 filters 
    tf.keras.layers.MaxPooling2D(), # picking the most prominent distinct features from our previous layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), # Flattens pooled map into a single vector for processing
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'), # makes accurate decision based on previous findings
    tf.keras.layers.Dense(len(class_names), activation='softmax') # turns vectors into a probability distroution 
])

#compile model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#train model
history = model.fit(data, validation_data = val_data ,epochs = 10) # 10 complete cycles through data

# Previe a batch of images
data_iterator = data.as_numpy_iterator()
batch = next(data_iterator)

# plot images from batch
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(batch[0][i])  # Show image
    plt.title(class_names[batch[1][i]])  # Show title as class name
    plt.axis("off")
plt.tight_layout()
plt.show()



def classify_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

with gr.Blocks() as demo:
    gr.Markdown("yo slime")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload an image")
        out = gr.Textbox(label="Predicted Class")
        callback=gr.CSVLogger()
    gr.Examples(
        examples=[
            ["testing-data/DC_202002_3832_OREOMcFlurry_1564x1564-1_product-header-mobile.jpeg"], 
            ["testing-data/DC_202302_0005-999_BigMac_1564x1564-1_product-header-mobile.jpeg"],
            ["testing-data/images (8).jpeg"],
            ["testing-data/images.jpeg"]

        ],
        inputs=inp
    )


    btn = gr.Button("Run")
    btn.click(fn=classify_image, inputs=inp, outputs=out)

    gr.ClearButton(inp, out)

    callback.setup([inp, out], "flagged_data_points")

    btn = gr.Button("Flag")
    btn.click(lambda *args: callback.flag(list(args)), [inp, out], None, preprocess=False)
    

demo.launch()





    


            

    