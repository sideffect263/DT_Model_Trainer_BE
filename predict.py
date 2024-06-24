import sys
import json
import joblib
import pandas as pd

def predict(features, data, session_id):
    try:
        # Load the trained model
        model_path = f'uploads/{session_id}/trained_model.pkl'
        model = joblib.load(model_path)
        
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data], columns=features)
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Return the predictions
        results = {'predictions': predictions.tolist()}
        print(json.dumps(results))
    
    except Exception as e:
        print("An error occurred:", str(e))
        results = {'error': str(e)}
        print(json.dumps(results))

if __name__ == "__main__":
    features = sys.argv[1].split(',')
    data = json.loads(sys.argv[2])
    session_id = sys.argv[3]

    predict(features, data, session_id)
