import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go   
import time
from predictcost import predict_price

# Set page configuration
st.set_page_config(
    page_title="final",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #ff4b4b, #f78500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-top: 1rem;
    }
    .subheader {
        font-size: 1.3rem;
        color: #1E88E5;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e03e3e, #e05e3e);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .feature-importance {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
        border-top: 1px solid #eee;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 6px rgba(255, 75, 75, 0.4);
    }
    .sidebar .css-1d391kg {
        padding: 2rem 1rem;
    }
    .sidebar-content {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-header">House Price Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 30px;">Get accurate estimates based on property features using advanced machine learning</p>', unsafe_allow_html=True)

# Sidebar for user info with improved styling
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    st.markdown("<h3 style='margin-top: 20px;'>About</h3>", unsafe_allow_html=True)
    st.info(
        """
        This application uses machine learning to predict house prices based on various property features and location data.
        """
    )
    st.markdown("<h3 style='margin-top: 20px;'>Instructions</h3>", unsafe_allow_html=True)
    st.success(
        """
        1. Adjust the property features
        2. Fill in the location information
        3. Click 'Calculate Price Estimate'
        """
    )
    
    st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Contact</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-content" style="text-align: center;">
        <a href="mailto:IamBatman@gothamcity.com">Sample@gmail.com</a><br>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🏠 Property Details", "📍 Location Information", "📊 Market Analysis"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Basic Property Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
        sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=1500, step=100)
        floors = st.slider("Number of Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    
    with col2:
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=8, value=2, step=1)
        sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=5000, step=500)
        condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
    
    with col3:
        waterfront = st.checkbox("Waterfront Property")
        view = st.slider("View Rating (0-4)", min_value=0, max_value=4, value=0)
        grade = st.slider("Grade (1-13)", min_value=1, max_value=13, value=7)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Additional Property Features</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sqft_above = st.number_input("Above Ground sqft", min_value=500, max_value=9000, value=1500, step=100)
        yr_built = st.slider("Year Built", min_value=1900, max_value=2025, value=1980)
    
    with col2:
        sqft_basement = st.number_input("Basement sqft", min_value=0, max_value=5000, value=0, step=50)
        yr_renovated = st.slider("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Location Details</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        zipcode = st.selectbox("Zipcode", 
                          options=[98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 
                                   98010, 98011, 98014, 98019, 98022, 98023, 98024, 98027,
                                   98028, 98029, 98030, 98031, 98032, 98033, 98034, 98038,
                                   98039, 98040, 98042, 98045, 98052, 98053, 98055, 98056,
                                   98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077,
                                   98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109,
                                   98112, 98115, 98116, 98117, 98118, 98119, 98122, 98125,
                                   98126, 98133, 98136, 98144, 98146, 98148, 98155, 98166,
                                   98168, 98177, 98178, 98188, 98198, 98199])
    
    with col2:
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            lat = st.number_input("Latitude", min_value=8.07, max_value=37.10, value=10.0, step=0.01)
        with col2_2:
            long = st.number_input("Longitude", min_value=68.12, max_value=97.35, value=78.02, step=0.01)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add map for location visualization
    st.markdown('<p class="subheader" style="margin-top: 20px;">Property Location</p>', unsafe_allow_html=True)
    
    # Create dataframe for the map
    map_data = pd.DataFrame({
        'lat': [lat],
        'lon': [long]
    })
    
    st.map(map_data, zoom=12)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Market Analysis</p>', unsafe_allow_html=True)

     # Create visualizations for market analysis
    market_col1, market_col2 = st.columns(2)
    
    with market_col1:
        # Price trends chart
        price_trend_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Price': [680000, 685000, 690000, 700000, 720000, 735000, 750000, 760000, 765000, 755000, 740000, 750000]
        })
        
        fig = px.line(
            price_trend_data, 
            x='Month', 
            y='Price',
            title='Average Home Price Trend (Last 12 Months)',
            markers=True
        )
        fig.update_traces(line=dict(color="#ff4b4b", width=3))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with market_col2:
        # Price per sqft by bedroom count
        price_per_sqft_data = pd.DataFrame({
            'Bedrooms': [1, 2, 3, 4, 5, 6],
            'Price_Per_SqFt': [280, 310, 330, 350, 370, 385]
        })
        
        fig = px.bar(
            price_per_sqft_data, 
            x='Bedrooms', 
            y='Price_Per_SqFt',
            title='Price per SqFt by Bedroom Count',
            color='Price_Per_SqFt',
            color_continuous_scale='Oranges',
            text_auto=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Prediction section
st.markdown('<div class="card" style="background-color: #f5f5f5; margin-top: 20px;">', unsafe_allow_html=True)
st.markdown('<p class="subheader" style="text-align: center;">Calculate Your House Price</p>', unsafe_allow_html=True)

predict_col1, predict_col2, predict_col3 = st.columns([2, 1, 2])
with predict_col2:
    predict_button = st.button("Calculate Price Estimate", type="primary")

if predict_button:
    # Create input data dictionary
    input_data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': 1 if waterfront else 0,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long
    }
    
    # Get prediction
    prediction = predict_price(input_data)
    
    # Animation effect for loading
    with st.spinner('Calculating price...'):
        import time
        time.sleep(1)  # Simulate processing time
    
    # Display the prediction with formatting
    st.markdown(f'<p class="prediction-text">Estimated Price: ₹{prediction * 85.45/9:,.2f}</p>', unsafe_allow_html=True)
    
    # Show feature importance in a nicer format
    st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
    st.markdown('<h4>Key factors affecting this prediction:</h4>', unsafe_allow_html=True)
    
    factors_col1, factors_col2 = st.columns(2)
    
    with factors_col1:
        st.markdown("""
        - **Location factors** (zipcode, lat/long)
        - **Property size** (living area, lot size)
        """)
    
    with factors_col2:
        st.markdown("""
        - **Quality metrics** (grade, condition)
        - **Amenities** (waterfront, view)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Property insights section
if predict_button:
    st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Property Insights</p>', unsafe_allow_html=True)
    
    # Create some sample insights based on the input
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        avg_price = prediction * 85.45 * 0.92  # Just for demonstration
        st.metric(label="Average Price in Area", value=f"₹{avg_price:,.2f}", delta=f"{(prediction * 85.45 - avg_price) / avg_price * 100:.1f}%")
        
    with insights_col2:
        renovation_impact = 0
        if yr_renovated == 0:
            renovation_impact = prediction * 85.45 * 0.08  # Just for demonstration
            st.metric(label="Potential Renovation Impact", value=f"₹{renovation_impact:,.2f}", delta="8%")
        else:
            st.metric(label="Recent Renovation Value", value=f"₹{prediction * 85.45 * 0.05:,.2f}", delta="5%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('House Price Prediction App | Built with Streamlit and Machine Learning', unsafe_allow_html=True)
st.markdown('© 2025 House Price Predictor', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)