import numpy as np
import pandas as pd
import os
from skimage import io, filters, exposure, util

# CONFIGURATION
CSV_PATH = 'C:/Users/m_lkn/OneDrive/Desktop/fractal-test/data/Data_Entry_2017_v2020.csv'
IMAGE_DIR = 'C:/Users/m_lkn/OneDrive/Desktop/fractal-test/sample_test/'
OUTPUT_CSV = 'lung_fractal_results.csv'

# FEATURE EXTRACTION FUNCTIONS

def get_fractal_dimension(img_binary):
    # Standard Box-Counting Algorithm
    p = int(np.floor(np.log2(min(img_binary.shape))))
    side = 2**p
    img = img_binary[:side, :side]
    
    scales = 2**np.arange(p, 2, -1)
    counts = []
    for s in scales:
        blocks = util.view_as_blocks(img, (s, s))
        count = np.sum(np.any(blocks, axis=(2, 3)))
        counts.append(count)
    
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    return coeffs[0]

def extract_features(image_path):
    # Loads image and returns Fractal Dimension & Lacunarity

    # 1. Load and enhance
    img = io.imread(image_path, as_gray=True)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    
    # 2. Binary Thresholding (Targeting lung tissue)
    thresh = filters.threshold_otsu(img_adapteq)
    binary = (img_adapteq > thresh).astype(np.uint8)
    
    # 3. Calculate Variables
    f_dim = get_fractal_dimension(binary)
    
    # Lacunarity (Texture consistency)
    s = 16 # Box size for texture
    h, w = binary.shape
    cropped = binary[:(h // s) * s, :(w // s) * s]
    blocks = util.view_as_blocks(cropped, (s, s))
    masses = np.sum(blocks, axis=(2, 3)).flatten()
    lac = (np.var(masses) / (np.mean(masses)**2)) + 1 if np.mean(masses) > 0 else 0
    
    return f_dim, lac

# MAIN PROCESSING LOOP

# Load metadata
df_meta = pd.read_csv(CSV_PATH)

processed_data = []

print(f"Starting analysis on {len(df_meta)} images...")

for index, row in df_meta.iterrows():
    img_name = row['Image Index']
    finding = row['Finding Labels']
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # Check if the file actually exists to avoid 'Image not found' errors
    if os.path.exists(img_path):
        try:
            f_dim, lac = extract_features(img_path)
            
            # Labeling Logic: Healthy (0) if No Finding, else Unhealthy (1)
            label = 0 if finding == "No Finding" else 1
            
            processed_data.append({
                'filename': img_name,
                'fractal_dim': f_dim,
                'lacunarity': lac,
                'label': label,
                'original_finding': finding
            })
            print(f"Processed {img_name} ({'Healthy' if label==0 else 'Unhealthy'})")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    else:
        print(f"Skip: {img_name} not found in folder.")

# Save results to a new CSV for model

results_df = pd.DataFrame(processed_data)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nFeatures saved to {OUTPUT_CSV}")