import pickle
import numpy as np
import pandas as pd

def load_model_and_scaler():
    """
    Load the trained model and scaler from pickle files
    """
    try:
        # Load the model
        with open('house_price_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
            
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found. Please run housepricemodel.py first.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_price(input_data):
    """
    Predict house price based on input features
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing feature values
        
    Returns:
    --------
    float
        Predicted house price
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return "Error: Model not loaded correctly"
    
    try:
        # Convert input data to DataFrame with correct feature order
        features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'
        ]
        
        # Create DataFrame with one row
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Keep only the required features and in the right order
        input_df = input_df[features]
        
        # Optional: Scale the input features (if scaler was used during training)
        # input_scaled = scaler.transform(input_df)
        
        # Make prediction (using unscaled input as RandomForest doesn't require scaling)
        prediction = model.predict(input_df)[0]
        
        return prediction
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft_living': 2000,
        'sqft_lot': 5000,
        'floors': 1,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 7,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'yr_built': 1990,
        'yr_renovated': 0,
        'zipcode': 98052,
        'lat': 47.6,
        'long': -122.2
    }
    
    predicted_price = predict_price(sample_input)
    print(f"Predicted house price: ${predicted_price:,.2f}")