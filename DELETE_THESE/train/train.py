from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1. Load your processed results (the CSV we generated earlier)
data = pd.read_csv('lung_fractal_results.csv')

# 2. Define Features (X) and Target (y)
X = data[['fractal_dim', 'lacunarity']]
y = data['label']

# 3. Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize and Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the SAME scaler for the test set

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. TEST the model's accuracy
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"Science Fair Result: The model is {accuracy * 100:.2f}% accurate on unseen images!")