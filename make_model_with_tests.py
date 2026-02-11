# Paths to training images folders
healthy_train_folder = "C:/Users/m_lkn/OneDrive/Desktop/fractal-test/healthy_train"
unhealthy_train_folder = "C:/Users/m_lkn/OneDrive/Desktop/fractal-test/unhealthy_train"

# Path to test images folder
test_images_folder = "C:/Users/m_lkn/OneDrive/Desktop/fractal-test/test_images"

# Allowed image formats
valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

import os
import csv
import numpy as np
from skimage import morphology as morph, io, filters, util
import pandas as pd
import pickle as pk
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc

# Be able to compute fractal metrics
def get_fractal_dimension(image_path):
    # Load image and convert to binary
    img = io.imread(image_path, as_gray=True)
    thresh = filters.threshold_otsu(img)
    binary = img < thresh
    binary = morph.skeletonize(binary)

    # Box-counting method
    def count_boxes(img, k):
        S = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0), np.arange(0, img.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    p = int(np.floor(np.log(min(binary.shape)) / np.log(2)))
    sizes = 2**np.arange(p, 1, -1)
    
    # Get counts for each size
    counts = []
    for s in sizes:
        counts.append(count_boxes(binary, s))
    
    # Fit line to log-log data, fractal dimension is negative slope
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
def get_lacunarity(image_path):
    # Load image and convert to binary
    img = io.imread(image_path, as_gray=True)
    thresh = filters.threshold_otsu(img)
    binary = img < thresh
    binary = morph.skeletonize(binary)

    # Lacunarity calculation
    p = int(np.floor(np.log(min(binary.shape)) / np.log(2)))
    sizes = 2 ** np.arange(p, 1, -1)
    lacunarity_values = []
    
    # Iterate over box sizes
    for s in sizes:
        # Create blocks
        h, w = binary.shape
        new_h = (h // s) * s
        new_w = (w // s) * s
        cropped_binary = binary[:new_h, :new_w]
        blocks = util.view_as_blocks(cropped_binary, (s, s))
        mass = np.sum(blocks, axis=(2, 3)).flatten()
        
        # Calculate lacunarity of box
        mean_mass = np.mean(mass)
        if mean_mass > 0:
            variance = np.var(mass)
            lac = (variance / (mean_mass ** 2)) + 1
            lacunarity_values.append(lac)
            
    return np.mean(lacunarity_values) if lacunarity_values else 0
def get_succolarity(image_path):
    # Load image and convert to binary
    img = io.imread(image_path, as_gray=True)
    thresh = filters.threshold_otsu(img)
    binary = img < thresh
    binary = morph.skeletonize(binary)

    # Initialize succolarity list
    succolarity_per_scale = []
    h, w = binary.shape
    max_box_size = min(h, w) // 4
    if max_box_size < 2: 
        max_box_size = 2
    box_sizes = np.linspace(2, max_box_size, 5, dtype=int)

    # Iterate over box sizes
    for s in box_sizes:
        reduced_h = h // s
        reduced_w = w // s
        occupancy = np.zeros((reduced_h, reduced_w))
        
        # Calculate occupancy for each box
        for i in range(reduced_h):
            for j in range(reduced_w):
                box = binary[i * s : (i + 1) * s, j * s : (j + 1) * s]
                occupancy[i, j] = np.sum(box) / (s * s)

        # Calculate succolarity for this scale
        pressure = np.tile(np.arange(1, reduced_h + 1).reshape(-1, 1), (1, reduced_w))
        numerator = np.sum(occupancy * pressure)
        denominator = np.sum(pressure)
        
        if denominator > 0:
            succolarity_per_scale.append(numerator / denominator)
    
    # Return average succolarity
    return np.mean(succolarity_per_scale) if succolarity_per_scale else 0.0

# Be able to process folders of training images
def process_folder(input_folder, output_csv, condition):
    # Prepare CSV file
    header = ["filename", "fractal_dimension", "lacunarity", "succolarity", "condition"]
    
    # Open CSV for writing
    with open(output_csv, mode='w', newline='') as f:
        # Create writer
        writer = csv.writer(f)
        writer.writerow(header)

        # Iterate through files
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_exts):
                print(f"Processing {filename}...", end = " ")
                image_path = os.path.join(input_folder, filename)
                
                # Calculate metrics
                try:
                    frac_dim = get_fractal_dimension(image_path)
                    lacuna = get_lacunarity(image_path)
                    succo = get_succolarity(image_path)
                    
                    # Write to CSV file
                    writer.writerow([filename, frac_dim, lacuna, succo, condition])
                    print("complete, moving to next image")
                except Exception as e:
                    print(f"error processing due to {e}")

# Process healthy images
process_folder(healthy_train_folder, "healthy-train_results.csv", 0)
print("Healthy training data processing complete.")

# Process unhealthy images
process_folder(unhealthy_train_folder, "unhealthy-train_results.csv", 1)
print("Unhealthy training data processing complete.")

# Merge healthy and unhealthy training data
healthy_train = pd.read_csv("healthy-train_results.csv")
unhealthy_train = pd.read_csv("unhealthy-train_results.csv")
training_data = pd.concat([healthy_train, unhealthy_train])
training_data.to_csv("all-train_results.csv", index=False)
print("Merging of healthy and unhealthy training data complete")

# Make the model
print("Open file ___ to create model")

# Definition to Make Model
def train_model():
    # Train model using all-train_results.csv
    data = pd.read_csv("all-train_results.csv")

    # Features and labels
    X = data[["fractal_dimension", "lacunarity", "succolarity"]]
    y = data["condition"]

    # Split data
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = rfc(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = acc(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save model
    with open("fractal_model.pkl", "wb") as f:
        pk.dump(model, f)
    print("Model saved as fractal_model.pkl")

# Make Model
train_model()

# Iterate through all files in test-images
def process_test_folder(test_folder, output_csv):
    header = ["filename", "fractal_dimension", "lacunarity", "succolarity", "prediction"]

    # Load model
    with open("fractal_model.pkl", "rb") as f:
        model = pk.load(f)

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filename in os.listdir(test_folder):
            if not filename.lower().endswith(valid_exts):
                continue
            image_path = os.path.join(test_folder, filename)
            print(f"Testing {filename}...", end=" ")
            try:
                # Calculate metrics for test image
                fd = get_fractal_dimension(image_path)
                lac = get_lacunarity(image_path)
                suc = get_succolarity(image_path)

                # Predict condition using model
                test_df = pd.DataFrame({
                    "fractal_dimension": [fd],
                    "lacunarity": [lac],
                    "succolarity": [suc]
                })
                pred = int(model.predict(test_df)[0])
                writer.writerow([filename, fd, lac, suc, pred])
                print(f"predicted {pred}")
            except Exception as e:
                print(f"error: {e}")
    
    print(f"Testing in {test_images_folder} complete, results saved to {output_csv}")

# Call function on your test_images folder
process_test_folder(test_images_folder, "test_results.csv")

# End of process_training_images.py