# Iterate though each line in Used_Data.csv and determine if the data contains "No Finding"
# If it does, add the data to healthy-train_results.csv, otherwise add it to unhealthy-train_results.csv

import csv
import numpy as np
from skimage import morphology as morph, io, filters, util
# To import these libraries, you may need to run one of the following:
# pip install scikit-image (to be permanently installed in your Python environment)
# c:\Users\m_lkn\OneDrive\Desktop\fractal-test\.venv\Scripts\python.exe -m pip install scikit-image

# To run this code, run the following command in the terminal:
# c:\Users\m_lkn\OneDrive\Desktop\fractal-test\.venv\Scripts\python.exe Used_Data_Running.py

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
    sizes = 2 ** np.arange(p, 1, -1)
    
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
    return np.mean(succolarity_per_scale) if succolarity_per_scale else 0

print("Starting processing of opened data...")
output_file = 'all_data_results_c.csv'
with open('Used_Data.csv', 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['image_path', 'fractal_dimension', 'lacunarity', 'succolarity', 'diagnosis', 'actual_condition'])
    for row in reader:
        abbr_path = row.get('Image Index')
        image_path = "D:\\Project Images\\" + abbr_path
        print(f"Beginning to [process] {image_path}...")
        label = row.get('Finding Labels', 'No Finding')
        fractal_dimension = get_fractal_dimension(image_path)
        lacunarity = get_lacunarity(image_path)
        succolarity = get_succolarity(image_path)
        diagnosis = 'healthy' if label == 'No Finding' else 'unhealthy'
        csv_writer.writerow([image_path, fractal_dimension, lacunarity, succolarity, diagnosis, diagnosis])
        print(f"Processing of {image_path} complete: Diagnosis={diagnosis} ; Fractal Dimension={fractal_dimension:.4f} ; Lacunarity={lacunarity:.4f} ; Succolarity={succolarity:.4f}")
print("Processing complete. Output saved to .csv file.")

# End of code