import numpy as np
from skimage import io, filters, util  # Changed util here
import os

def get_lacunarity(image_path):
    # 1. Load and Preprocess
    img = io.imread(image_path, as_gray=True)
    thresh = filters.threshold_otsu(img)
    binary = (img < thresh).astype(np.uint8) # Convert to 0s and 1s
    
    p = int(np.floor(np.log(min(binary.shape)) / np.log(2)))
    sizes = 2**np.arange(p, 1, -1)
    
    lacunarity_values = []
    
    for size in sizes:
        # 2. Reshape/Crop image to be divisible by the box size
        # This prevents the AttributeError/Shape mismatch
        h, w = binary.shape
        new_h = (h // size) * size
        new_w = (w // size) * size
        cropped_binary = binary[:new_h, :new_w]
        
        # 3. Use util.view_as_blocks (The fix for your error)
        blocks = util.view_as_blocks(cropped_binary, (size, size))
        
        # Calculate mass (number of pixels) in each block
        mass = np.sum(blocks, axis=(2, 3)).flatten()
        
        # 4. Lacunarity calculation
        mean_mass = np.mean(mass)
        if mean_mass > 0:
            # Lacunarity = (Variance / Mean^2) + 1
            # Or simplified: E[M^2] / (E[M])^2
            variance = np.var(mass)
            lac = (variance / (mean_mass**2)) + 1
            lacunarity_values.append(lac)
            
    return np.mean(lacunarity_values) if lacunarity_values else 0

print(get_lacunarity(image_path='C:/Users/m_lkn/OneDrive/Desktop/fractal-test/images/00006585_008.png'))