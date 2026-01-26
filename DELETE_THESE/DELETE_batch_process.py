import numpy as np
import pandas as pd
import os
from skimage import io, filters, exposure, util
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_fractal_dimension_box_counting(img_binary):
    """
    Standard Box-Counting Algorithm.
    Counts how many boxes of size 's' contain at least one foreground pixel.
    """
    # Ensure image is square and power of 2 for the cleanest box-counting
    p = int(np.floor(np.log2(min(img_binary.shape))))
    side = 2**p
    img = img_binary[:side, :side]
    
    scales = 2**np.arange(p, 2, -1) # Box sizes: 256, 128, 64, 32...
    counts = []
    
    for s in scales:
        # Reshape into blocks of size s x s and count non-zero blocks
        # This is the core of box-counting
        blocks = util.view_as_blocks(img, (s, s))
        # Count blocks that contain at least one '1' (lung structure)
        count = np.sum(np.any(blocks, axis=(2, 3)))
        counts.append(count)
    
    # The Fractal Dimension is the slope of log(counts) vs log(1/scales)
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    return coeffs[0]

def process_image(path):
    # 1. Load and improve contrast (CLAHE)
    img = io.imread(path, as_gray=True)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    
    # 2. Binary Thresholding
    thresh = filters.threshold_otsu(img_adapteq)
    binary = (img_adapteq > thresh).astype(np.uint8)
    
    # 3. Extract Fractal Dimension using Box Counting
    f_dim = get_fractal_dimension_box_counting(binary)
    
    # 4. Extract Lacunarity (Texture 'gappiness')
    # We'll use a fixed box size of 16 for a consistent texture variable
    s = 16
    h, w = binary.shape
    cropped = binary[:(h // s) * s, :(w // s) * s]
    blocks = util.view_as_blocks(cropped, (s, s))
    masses = np.sum(blocks, axis=(2, 3)).flatten()
    lac = (np.var(masses) / (np.mean(masses)**2)) + 1 if np.mean(masses) > 0 else 0
    
    return f_dim, lac

# --- RUNNING THE MODEL ---

# 1. Provide your training examples here
# (At least 2 healthy and 2 unhealthy to avoid the 'all healthy' error)
# Format: [Fractal_Dim, Lacunarity]
X_train = np.array([
    [1.3977031264603899, 2.088287076679679], # Example Unhealthy 1
    [1.4178934486810513, 1.8799656480521871], # Example Unhealthy 2
    [1.3980029784147643, 1.4966023821859902], # Example Healthy 1
    [1.483246258475107, 2.2229455068137742]  # Example Healthy 2
])
y_train = np.array([1, 1, 0, 0]) # 1=Unhealthy, 0=Healthy

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_scaled, y_train)

# 2. Test a new image
test_path = 'C:/Users/m_lkn/OneDrive/Desktop/fractal-test/images/00006586_000.png'

if os.path.exists(test_path):
    f, l = process_image(test_path)
    new_data = scaler.transform([[f, l]])
    prediction = clf.predict(new_data)[0]
    
    status = "UNHEALTHY" if prediction == 1 else "HEALTHY"
    print(f"File: {os.path.basename(test_path)}")
    print(f"Fractal Dimension: {f:.4f} | Lacunarity: {l:.4f}")
    print(f"Model Result: {status}")
else:
    print("Image not found.")