import numpy as np
import pandas as pd
import os
from skimage import io, filters, util, morphology
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- 1. FEATURE EXTRACTION FUNCTIONS ---

def extract_features(image_path):
    """Extracts Fractal Dimension and Lacunarity from an image."""
    try:
        img = io.imread(image_path, as_gray=True)
        # Thresholding to binary (lung structures)
        thresh = filters.threshold_otsu(img)
        binary = (img < thresh).astype(np.uint8)
        
        # Fractal Dimension (Box Counting)
        p = int(np.floor(np.log(min(binary.shape)) / np.log(2)))
        sizes = 2**np.arange(p, 1, -1)
        counts = []
        for size in sizes:
            S = np.add.reduceat(np.add.reduceat(binary, np.arange(0, binary.shape[0], size), axis=0),
                                 np.arange(0, binary.shape[1], size), axis=1)
            counts.append(len(np.where((S > 0) & (S < size*size))[0]))
        
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        df_val = -coeffs[0]

        # Lacunarity
        lac_values = []
        for size in sizes:
            h, w = binary.shape
            cropped = binary[:(h // size) * size, :(w // size) * size]
            blocks = util.view_as_blocks(cropped, (size, size))
            mass = np.sum(blocks, axis=(2, 3)).flatten()
            if np.mean(mass) > 0:
                lac = (np.var(mass) / (np.mean(mass)**2)) + 1
                lac_values.append(lac)
        
        lac_val = np.mean(lac_values) if lac_values else 0
        return [df_val, lac_val]
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return [0, 0]

# --- 2. MODEL TRAINING (SIMULATED) ---

# In a real project, you would loop through a folder of 50+ images to fill this
# For now, we use your known samples to 'teach' the model
data = {
    'fractal_dim': [1.75, 1.60, 1.78, 1.55], # Higher usually means more 'cloudy' (unhealthy)
    'lacunarity': [0.45, 0.21, 0.48, 0.19],  # Higher usually means more irregular
    'label': [1, 0, 1, 0] # 1 = Unhealthy, 0 = Healthy
}

train_df = pd.DataFrame(data)
X_train = train_df[['fractal_dim', 'lacunarity']]
y_train = train_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- 3. THE PREDICTION INTERFACE ---

def classify_lung(image_path):
    print(f"\nAnalyzing: {os.path.basename(image_path)}...")
    
    # Get variables
    features = extract_features(image_path)
    print(f"Extracted -> Fractal Dimension: {features[0]:.4f}, Lacunarity: {features[1]:.4f}")
    
    # Scale and Predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    result = "UNHEALTHY" if prediction == 1 else "HEALTHY"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    print(f"RESULT: {result} ({confidence*100:.1f}% confidence)")

# --- 4. EXECUTION ---
# Replace these with your actual local paths
test_image = "C:/Users/m_lkn/OneDrive/Desktop/fractal-test/images/00006586_004.png"

if os.path.exists(test_image):
    classify_lung(test_image)
else:
    print("File not found. Please check your path!")