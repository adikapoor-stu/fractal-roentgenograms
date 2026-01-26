import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
import os
# import pandas as pd

from pathlib import Path

image_length = 100

# Generate Dataset
def generate_mandelbrot(max_iter=100):
    img = np.zeros((image_length, image_length), dtype=np.uint8)
    for x in range(image_length):
        for y in range(image_length):
            c = complex(-2 + x * 3 / image_length, -1.5 + y * 3 / image_length)
            z = 0
            for i in range(max_iter):
                z = z * z + c
                if abs(z) > 2:
                    img[y, x] = i
                    break
    return img

def generate_julia_set(c, max_iter=100):
    img = np.zeros((image_length, image_length), dtype=np.uint8)
    for x in range(image_length):
        for y in range(image_length):
            z = complex(-1.5 + x * 3 / image_length, -1.5 + y * 3 / image_length)
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    img[y, x] = i
                    break
    return img

def create_dataset(base_dir='fractal_dataset', num_images=100):
    # IMPORTANT: make the bottom 2 lines comments if the folders already exist, otherwise uncomment
    # os.makedirs(os.path.join(base_dir, 'fractal'))
    # os.makedirs(os.path.join(base_dir, 'non_fractal'))

    for i in range(num_images):
        
        # Generate Fractal Images (Mandelbrot and Julia)

        mandelbrot_img = generate_mandelbrot(max_iter=50)
        img_mandelbrot = Image.fromarray(mandelbrot_img * (255 // 50)).convert('RGB')
        img_mandelbrot.save(os.path.join(base_dir, 'fractal', f'mandelbrot_{i}.png'))

        c = complex(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        julia_set = generate_julia_set(c, max_iter=50)
        img_julia = Image.fromarray(julia_set * (255 // 50)).convert('RGB')
        img_julia.save(os.path.join(base_dir, 'fractal', f'julia_{i}.png'))

        # Generate Non-Fractal Images (random noise)
        noise = np.random.randint(0, 256, size=(image_length, image_length, 3), dtype=np.uint8)
        img_noise = Image.fromarray(noise)
        img_noise.save(os.path.join(base_dir, 'non_fractal', f'noise_{i}.png'))
        
    print("Dataset created successfully.")


# Build and Train CNN
def train_fractal_detector():
    batch_size = 32

    # Load data with tensorflow
    dataset_dir = 'fractal_dataset'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_length, image_length),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_length, image_length),
        batch_size=batch_size
    )

    # Normalize pixel values to [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Build CNN model
    num_classes = 2
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(image_length, image_length, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    print(model.summary())

    # Train model
    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Save trained model
    model.save('fractal_detector_model.keras')
    print("Model saved as fractal_detector_model.keras")

    return model

# Use model to predict
def predict_fractal(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(image_length, image_length))
    img_array = tf.keras.utils.img_to_array(img)
    # Create a batch
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['fractal', 'non_fractal']
    confidence = 100 * np.max(score)
    
    
    print(f"Image '{image_path}' is most likely a '{class_names[np.argmax(score)]}' with {confidence:.2f}% confidence.")

# Execution
if __name__ == "__main__":
    # Create da dataset (line 38)
    create_dataset()

    # Train the model
    trained_model = train_fractal_detector()

    # Test an image
    path = Path('./images')
    image_count = len(list(path.iterdir()))
    for current_image in range (image_count):
        test_image_path = os.path.join(path, os.listdir(path)[current_image])
        # example: test_image_path = 'images/00006585_007.png'
        predict_fractal(test_image_path, trained_model)