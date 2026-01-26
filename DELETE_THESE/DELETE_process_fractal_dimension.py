import numpy as np
import cv2
from skimage import io, color, filters, morphology

def get_fractal_dimension(image_path):
    # 1. Load and Preprocess
    img = io.imread(image_path, as_gray=True)
    
    # Thresholding to create a binary mask of the lung structures
    # Fractals are usually calculated on binary (black/white) images
    thresh = filters.threshold_otsu(img)
    binary = img < thresh  # Inverting depending on X-ray contrast
    binary = morphology.skeletonize(binary) # Reduces structures to 1-pixel width

    # 2. Box Counting Algorithm
    def count_boxes(img, k):
        # Sums up blocks of size k by k
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                                 np.arange(0, img.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Determine the range of box sizes (powers of 2)
    p = int(np.floor(np.log(min(binary.shape)) / np.log(2)))
    sizes = 2**np.arange(p, 1, -1)
    
    counts = []
    for size in sizes:
        counts.append(count_boxes(binary, size))
    
    # 3. Linear Regression to find the slope (Fractal Dimension)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0] # The negative slope is the Fractal Dimension (D)

# Example usage:
d_value = get_fractal_dimension('C:/Users/m_lkn/OneDrive/Desktop/fractal-test/images/00006585_008.png')
print(f"Fractal Dimension: {d_value}")