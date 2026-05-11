import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .prediction-result {
        font-size: 2rem;
        color: #2E86AB;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #e8f4fd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate synthetic housing data
@st.cache_data
def generate_housing_data(n_samples=1000):
    """Generate synthetic housing dataset for demonstration"""
    np.random.seed(42)
    
    # Define locations and their price multipliers
    locations = ['Downtown', 'Suburb', 'Waterfront', 'Historic District', 'Industrial']
    location_multipliers = [1.5, 1.0, 2.0, 1.3, 0.8]
    
    # Generate features
    data = {
        'area': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.choice([2, 3, 4, 5, 6], n_samples, p=[0.15, 0.3, 0.35, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
        'age': np.random.randint(0, 50, n_samples),
        'location': np.random.choice(locations, n_samples),
        'garage': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2]),
        'garden': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Ensure area is positive
    df['area'] = np.abs(df['area'])
    df['area'] = df['area'].clip(lower=800, upper=5000)
    
    # Create price based on features with some noise
    base_price = 100000
    price_per_sqft = 150
    
    df['price'] = (
        base_price +
        df['area'] * price_per_sqft +
        df['bedrooms'] * 15000 +
        df['bathrooms'] * 10000 +
        (50 - df['age']) * 2000 +  # Newer houses cost more
        df['garage'] * 20000 +
        df['garden'] * 15000 +
        np.random.normal(0, 50000, n_samples)  # Add noise
    )
    
    # Apply location multiplier
    for i, location in enumerate(locations):
        df.loc[df['location'] == location, 'price'] *= location_multipliers[i]
    
    # Ensure positive prices
    df['price'] = np.abs(df['price'])
    df['price'] = df['price'].round(-3)  # Round to nearest thousand
    
    return df

# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    """Preprocess the housing data"""
    # Handle missing values (if any)
    df = df.dropna()
    
    # Encode categorical variables
    le_location = LabelEncoder()
    df_processed = df.copy()
    df_processed['location_encoded'] = le_location.fit_transform(df['location'])
    
    return df_processed, le_location

# Function to train model
@st.cache_data
def train_model(df_processed):
    """Train the Linear Regression model"""
    # Prepare features and target
    features = ['area', 'bedrooms', 'bathrooms', 'age', 'location_encoded', 'garage', 'garden']
    X = df_processed[features]
    y = df_processed['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    return model, X_test, y_test, y_pred, metrics, features

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and preprocess the housing data"""
    df = generate_housing_data(1000)
    df_processed, le_location = preprocess_data(df)
    return df, df_processed, le_location

# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 House Price Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("### *Powered by Linear Regression & Data Science*")
    
    # Load data
    df, df_processed, le_location = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["📊 Exploratory Data Analysis", 
                                "🔍 Model Performance", 
                                "🏡 Price Prediction"])
    
    # Train model
    model, X_test, y_test, y_pred, metrics, features = train_model(df_processed)
    
    if page == "📊 Exploratory Data Analysis":
        st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Houses", len(df))
        with col2:
            st.metric("Average Price", f"${df['price'].mean():,.0f}")
        with col3:
            st.metric("Average Area", f"{df['area'].mean():.0f} sq ft")
        with col4:
            st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        # Display sample data
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(10))
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Feature Correlations", "Location Analysis", "Feature Relationships"])
        
        with tab1:
            fig = px.histogram(df, x='price', nbins=50, title='House Price Distribution')
            fig.update_layout(xaxis_title='Price ($)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True)
            
            # Price statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Price Statistics:**")
                st.write(df['price'].describe())
            with col2:
                # Box plot
                fig_box = px.box(df, y='price', title='Price Distribution (Box Plot)')
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab2:
            # Correlation heatmap
            numeric_cols = ['area', 'bedrooms', 'bathrooms', 'age', 'garage', 'garden', 'price']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Location analysis
            location_stats = df.groupby('location')['price'].agg(['mean', 'count']).reset_index()
            location_stats.columns = ['Location', 'Average Price', 'Count']
            
            fig = px.bar(location_stats, x='Location', y='Average Price', 
                        title='Average House Price by Location')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(location_stats)
        
        with tab4:
            # Feature relationships
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(df, x='area', y='price', color='location',
                               title='Price vs Area by Location')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='age', y='price', color='bedrooms',
                               title='Price vs Age by Bedrooms')
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Model Performance":
        st.markdown('<h2 class="sub-header">🔍 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{metrics['R²']:.3f}")
        with col2:
            st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
        with col3:
            st.metric("MAE", f"${metrics['MAE']:,.0f}")
        with col4:
            st.metric("MSE", f"{metrics['MSE']:,.0f}")
        
        # Performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = px.scatter(x=y_test, y=y_pred, 
                           title='Actual vs Predicted Prices',
                           labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'})
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                         line=dict(color="red", dash="dash"))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred
            fig = px.scatter(x=y_pred, y=residuals,
                           title='Residuals Plot',
                           labels={'x': 'Predicted Price ($)', 'y': 'Residuals ($)'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        feature_names = ['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location', 'Garage', 'Garden']
        importance = np.abs(model.coef_)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h', title='Feature Importance (Absolute Coefficients)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Model equation
        st.subheader("📐 Model Equation")
        equation = f"**Price = {model.intercept_:,.0f}"
        for i, (feature, coef) in enumerate(zip(feature_names, model.coef_)):
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:,.0f} × {feature}"
        equation += "**"
        st.markdown(equation)
    
    elif page == "🏡 Price Prediction":
        st.markdown('<h2 class="sub-header">🏡 House Price Prediction</h2>', unsafe_allow_html=True)
        st.write("Enter the house features below to get an estimated price:")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("🏠 Area (sq ft)", min_value=500, max_value=5000, value=2000, step=50)
            bedrooms = st.selectbox("🛏️ Bedrooms", [2, 3, 4, 5, 6], index=1)
            bathrooms = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4], index=1)
            age = st.slider("📅 Age (years)", 0, 50, 10)
        
        with col2:
            location = st.selectbox("📍 Location", df['location'].unique())
            garage = st.selectbox("🚗 Garage Spaces", [0, 1, 2], index=1)
            garden = st.selectbox("🌿 Garden", ['No', 'Yes'], index=1)
        
        # Convert inputs
        location_encoded = le_location.transform([location])[0]
        garden_encoded = 1 if garden == 'Yes' else 0
        
        # Predict button
        if st.button("🔮 Predict Price", type="primary"):
            # Prepare input data
            input_data = np.array([[area, bedrooms, bathrooms, age, location_encoded, garage, garden_encoded]])
            
            # Make prediction
            predicted_price = model.predict(input_data)[0]
            
            # Display result
            st.markdown(f'<div class="prediction-result">💰 Estimated Price: ${predicted_price:,.0f}</div>', 
                       unsafe_allow_html=True)
            
            # Price breakdown
            st.subheader("📊 Price Breakdown")
            
            # Calculate contribution of each feature
            base_contribution = model.intercept_
            feature_contributions = model.coef_ * input_data[0]
            
            breakdown_data = {
                'Component': ['Base Price'] + feature_names,
                'Contribution': [base_contribution] + list(feature_contributions),
                'Value': ['N/A', f"{area} sq ft", f"{bedrooms}", f"{bathrooms}", 
                         f"{age} years", location, f"{garage}", garden]
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            breakdown_df['Contribution'] = breakdown_df['Contribution'].round(0)
            
            st.dataframe(breakdown_df)
            
            # Price comparison
            st.subheader("📈 Price Comparison")
            
            location_avg = df[df['location'] == location]['price'].mean()
            overall_avg = df['price'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Estimate", f"${predicted_price:,.0f}")
            with col2:
                st.metric(f"{location} Average", f"${location_avg:,.0f}")
            with col3:
                st.metric("Overall Average", f"${overall_avg:,.0f}")

if __name__ == "__main__":
    main()