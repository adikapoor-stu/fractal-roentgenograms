import pickle as pk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc

def testImage():
    # Load model
    with open("fractal_model.pkl", "rb") as f:
        model = pk.load(f)

    # Example test data (replace with actual image metrics)
    test_data = pd.DataFrame({
        "fractal_dimension": [1.5],
        "lacunarity": [0.3],
        "succolarity": [0.2]
    })

    # Predict condition
    prediction = model.predict(test_data)
    print(f"Predicted condition: {'Unhealthy' if prediction[0] == 1 else 'Healthy'}")


# End of train_model.py