import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

def build_model():
    """
    Build and train the house price prediction model
    """
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/kc_house_data.csv')
    
    # Basic exploration
    print(f"Dataset shape: {df.shape}")
    print("First few rows:")
    print(df.head())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Feature selection
    print("\nSelecting features...")
    selected_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'
    ]
    
    X = df[selected_features]
    y = df['price']
    
    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest model
    print("Training RandomForest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    importance = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")
    return rf_model, scaler

if __name__ == "__main__":
    build_model()