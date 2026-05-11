import streamlit as st
import pandas as pd
import numpy as np
import pickle
from predictcost import predict_price
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with more sophisticated styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #ff4b4b, #ff8f70);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-top: 1rem;
        font-weight: 700;
    }
    
    /* Card Styles */
    .card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 24px;
        border: 1px solid #eaeaea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    /* Section Headers */
    .subheader {
        font-size: 1.3rem;
        color: #1E88E5;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        position: relative;
        padding-left: 12px;
    }
    .subheader:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(180deg, #1E88E5, #64B5F6);
        border-radius: 4px;
    }
    
    /* Prediction Result Styles */
    .prediction-text {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #2e7d32, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Feature Importance Panel */
    .feature-importance {
        background-color: #e8f5e9;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid #2e7d32;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 6px 6px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff7043);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2.5rem;
        font-size: 1.1rem;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e03e3e, #e05e3e);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 6px rgba(255, 75, 75, 0.4);
    }
    
    /* Slider Customization */
    .stSlider [data-baseweb="slider"] {
        height: 5px;
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #ff4b4b;
        height: 20px;
        width: 20px;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        font-size: 0.9rem;
        color: #666;
        border-top: 1px solid #eee;
        background-color: #f9f9f9;
        border-radius: 0 0 10px 10px;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Chart Container */
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-top: 10px;
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for visualizations - you'd replace this with your actual data
@st.cache_data
def load_sample_data():
    # This is sample data - in a real app, you'd load your actual dataset
    data = {
        'zipcode': [98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008],
        'avg_price': [450000, 520000, 680000, 1200000, 980000, 750000, 840000, 920000],
        'price_per_sqft': [250, 290, 320, 510, 460, 380, 410, 430],
        'bedrooms': [3, 4, 3, 5, 4, 4, 3, 4],
        'avg_sqft': [1800, 1900, 2100, 2400, 2200, 2000, 2100, 2150]
    }
    return pd.DataFrame(data)

sample_data = load_sample_data()

# Helper function to generate comparable properties
def generate_comparable_properties(input_data, prediction):
    # In a real app, you'd query similar properties from your dataset
    # This is just a simple simulation
    comparable = []
    for i in range(5):
        # Simulate similar properties with variations
        comparable.append({
            'price': prediction * 85.45 * np.random.uniform(0.9, 1.1),
            'bedrooms': max(1, input_data['bedrooms'] + np.random.randint(-1, 2)),
            'bathrooms': max(1, input_data['bathrooms'] + np.random.randint(-1, 2)),
            'sqft': input_data['sqft_living'] * np.random.uniform(0.9, 1.1),
            'year_built': input_data['yr_built'] + np.random.randint(-10, 10),
            'distance': round(np.random.uniform(0.5, 3.0), 1)  # miles away
        })
    return comparable

# Main title with custom styling
st.markdown('<h1 class="main-header">House Price Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 30px;">Get accurate estimates based on property features using advanced machine learning</p>', unsafe_allow_html=True)

# Sidebar for user info with improved styling
with st.sidebar:
    # Logo and title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/home.png", width=60)
    with col2:
        st.markdown("<h2 style='margin-top: 12px;'>House Price</h2>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
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
    
    # Sample price distribution by zipcode chart in sidebar
    st.markdown("<h3 style='margin-top: 20px;'>Price Trends</h3>", unsafe_allow_html=True)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    fig = px.bar(
        sample_data, 
        x='zipcode', 
        y='avg_price',
        title='Average Price by Zipcode',
        color='avg_price',
        color_continuous_scale='Reds',
        labels={'avg_price': 'Average Price (₹)', 'zipcode': 'Zipcode'}
    )
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Contact info
    st.markdown("<h3 style='margin-top: 20px;'>Contact</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
        <a href="mailto:contact@example.com">contact@example.com</a><br>
        <a href="https://example.com" target="_blank">www.example.com</a>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Create tabs for better organization with icons
tab1, tab2, tab3 = st.tabs(["🏠 Property Details", "📍 Location Information", "📊 Market Analysis"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Basic Property Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3, help="Select the number of bedrooms in the property")
        sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=1500, step=100, help="Total square footage of the living area")
        floors = st.slider("Number of Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5, help="Number of floors in the property")
    
    with col2:
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=8, value=2, step=1, help="Select the number of bathrooms")
        sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=5000, step=500, help="Total square footage of the land")
        condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3, help="Overall condition of the property (1=Poor, 5=Excellent)")
    
    with col3:
        waterfront = st.checkbox("Waterfront Property", help="Check if the property has a waterfront view")
        view = st.slider("View Rating (0-4)", min_value=0, max_value=4, value=0, help="Quality of view (0=None, 4=Excellent)")
        grade = st.slider("Grade (1-13)", min_value=1, max_value=13, value=7, help="Overall grade given by King County grading system")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Additional Property Features</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sqft_above = st.number_input("Above Ground sqft", min_value=500, max_value=9000, value=1500, step=100, help="Square footage of house above ground level")
        yr_built = st.slider("Year Built", min_value=1900, max_value=2025, value=1980, help="Year the property was built")
    
    with col2:
        sqft_basement = st.number_input("Basement sqft", min_value=0, max_value=5000, value=0, step=50, help="Square footage of the basement")
        yr_renovated = st.slider("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0, help="Year of last renovation (0 if never renovated)")
    
    
    # Configure layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
            lat = st.number_input("Latitude", min_value=1.0, max_value=48.0, value=19.03, step=0.01)
        with col2_2:
            long = st.number_input("Longitude", min_value=-123.0, max_value=121.0, value=73.02, step=0.01)
    
    # Add map for location visualization
    st.markdown('<p class="subheader" style="margin-top: 20px;">Property Location</p>', unsafe_allow_html=True)
    
    # Create dataframe for the map
    map_data = pd.DataFrame({
        'lat': [lat],
        'lon': [long]
    })
    
    st.map(map_data, zoom=12)
    
    # Add neighborhood info
    st.markdown('<p class="subheader" style="margin-top: 20px;">Neighborhood Information</p>', unsafe_allow_html=True)
    
    neighborhood_col1, neighborhood_col2 = st.columns(2)
    
    with neighborhood_col1:
        st.metric("Median Home Value", f"₹{650000 * 85.45:,.2f}", "8.5%")
        st.metric("Crime Rate", "Low", "-5.2%")
    
    with neighborhood_col2:
        st.metric("School Rating", "8.3/10", "Top 15%")
        st.metric("Walk Score", "72/100", "Very Walkable")
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Add market insights
    st.markdown('<p class="subheader" style="margin-top: 10px;">Market Insights</p>', unsafe_allow_html=True)
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        st.metric("Median Days on Market", "18", "-22%")
    
    with insights_col2:
        st.metric("Sale-to-List Ratio", "102%", "3.1%")
    
    with insights_col3:
        st.metric("Inventory Change", "+156 listings", "12.3%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
st.markdown('<div class="card" style="background-color: #f5f5f5; margin-top: 20px;">', unsafe_allow_html=True)
st.markdown('<p class="subheader" style="text-align: center;">Calculate Your House Price</p>', unsafe_allow_html=True)

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
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
    
    # Animation effect for loading
    progress_text = "Analyzing property data..."
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # Simulate processing time
        progress_bar.progress(percent_complete + 1)
    
    # Get prediction
    prediction = predict_price(input_data)
    inr_prediction = prediction * 85.45
    
    # Display the prediction with formatting
    st.markdown(f'<p class="prediction-text">Estimated Price: ₹{inr_prediction:,.2f}</p>', unsafe_allow_html=True)
    
    # Show feature importance in a nicer format
    st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
    st.markdown('<h4>Key factors affecting this prediction:</h4>', unsafe_allow_html=True)
    
    # Create gauge chart for influence factors
    factors = ['Location', 'Size', 'Condition', 'Age']
    factor_values = [0.85, 0.76, 0.62, 0.45]  # These would be calculated from your model
    
    fig = go.Figure()
    
    for i, (factor, value) in enumerate(zip(factors, factor_values)):
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = value * 100,
            domain = {'row': 0, 'column': i},
            title = {'text': factor},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(255,75,75,0.8)"},
                'steps': [
                    {'range': [0, 33], 'color': "rgba(255,75,75,0.15)"},
                    {'range': [33, 66], 'color': "rgba(255,75,75,0.3)"},
                    {'range': [66, 100], 'color': "rgba(255,75,75,0.5)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value * 100
                }
            }
        ))
    
    fig.update_layout(
        grid = {'rows': 1, 'columns': len(factors), 'pattern': "independent"},
        height = 220,
        margin = {'t': 25, 'b': 25, 'l': 25, 'r': 25}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Property value components
    st.markdown('<p class="subheader" style="margin-top: 20px;">Value Components</p>', unsafe_allow_html=True)
    
    # Calculate sample value components - in a real app, these would be derived from your model
    land_value = inr_prediction * 0.35
    structure_value = inr_prediction * 0.50
    location_premium = inr_prediction * 0.15
    
    # Create value components chart
    value_data = pd.DataFrame({
        'Component': ['Land Value', 'Structure Value', 'Location Premium'],
        'Value': [land_value, structure_value, location_premium]
    })
    
    fig = px.pie(
        value_data, 
        values='Value', 
        names='Component',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    )
    fig.update_layout(height=300)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show comparable properties
    st.markdown('<p class="subheader" style="margin-top: 20px;">Comparable Properties</p>', unsafe_allow_html=True)
    
    comparables = generate_comparable_properties(input_data, prediction)
    
    # Create a dataframe for the comparables
    comp_df = pd.DataFrame(comparables)
    comp_df['price_formatted'] = comp_df['price'].apply(lambda x: f"₹{x:,.2f}")
    
    # Format the comparables table
    st.dataframe(
        comp_df[['price_formatted', 'bedrooms', 'bathrooms', 'sqft', 'year_built', 'distance']].rename(
            columns={
                'price_formatted': 'Price', 
                'bedrooms': 'Beds', 
                'bathrooms': 'Baths', 
                'sqft': 'Sq.ft', 
                'year_built': 'Year', 
                'distance': 'Distance (mi)'
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# Mortgage calculator section
st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
st.markdown('<p class="subheader">Mortgage Calculator</p>', unsafe_allow_html=True)

mortgage_col1, mortgage_col2 = st.columns(2)

with mortgage_col1:
    if predict_button:
        loan_amount = st.number_input("Loan Amount", min_value=100000.0, max_value=10000000.0, value=float(inr_prediction * 0.8), step=100000.0, format="%0.2f")
    else:
        loan_amount = st.number_input("Loan Amount", min_value=100000.0, max_value=10000000.0, value=2000000.0, step=100000.0, format="%0.2f")      