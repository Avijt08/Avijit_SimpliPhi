"""
Property RAG System - Enhanced Frontend
A beautiful and interactive Streamlit app for property search and analytics
"""
import re
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.figure import Figure
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# Set page configuration with custom theme
st.set_page_config(
    page_title="🏠 Property Search",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
    }
    
    /* Header styling */
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Card styling */
    .property-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .property-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    
    /* Tab styling */
    .stTab {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Custom header with gradient */
    .gradient-header {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C8C 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Feature badges */
    .feature-badge {
        display: inline-block;
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        border: 1px solid #e9ecef;
    }
    
    /* Price tag */
    .price-tag {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C8C 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Insights card */
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #FF4B4B;
    }
    
    /* Custom sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Custom metric tiles */
    .metric-tile {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .loading {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = None

# Custom header with animation
st.markdown("""
    <div class="gradient-header">
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🏠 Smart Property Search</h1>
        <p style='font-size: 1.2rem; opacity: 0.9;'>Discover Your Perfect Home with AI</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs with custom styling
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Analytics", "ℹ️ Guide"])

def create_search_form():
    """Creates and handles the search form inputs - returns query and filter parameters"""
    query = None
    price_min = None
    price_max = None
    property_type = None
    
    input_type = st.radio(
        "Choose your search method:",
        ["Natural Question", "Price Filter", "Custom Search"],
        key="input_type",
        help="Natural Question: Ask anything about properties\nPrice Filter: Search by price range\nCustom Search: Use specific filters",
        horizontal=True
    )

    if input_type == "Natural Question":
        query = st.text_input("Enter your property question:", 
                          placeholder="E.g., What's the average price of 3-bedroom homes?",
                          key="query_input",
                          label_visibility="collapsed")
    
    elif input_type == "Price Filter":
        filter_type = st.radio(
            "Price filter type:",
            ["Exact price", "Maximum price", "Price range"],
            key="price_filter_type"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_price_input = st.number_input("Minimum price (£):", min_value=0, value=0, step=100) if filter_type == "Price range" else 0
            price_input = st.number_input(
                "Price (£):" if filter_type == "Exact price" else "Maximum price (£):",
                min_value=0, value=0, step=100
            )
        
        with col2:
            margin = st.slider("Price margin (%)", 0, 50, 10) if filter_type == "Exact price" else 0
            property_type = st.selectbox(
                "Property type (optional):",
                ["", "Apartment", "Flat", "Terraced", "Detached", "Semi-detached", "Studio"],
                index=0
            )
        
        if price_input > 0:
            if filter_type == "Exact price":
                margin_amount = price_input * (margin / 100)
                price_min = max(0, price_input - margin_amount)
                price_max = price_input + margin_amount
                query = f"Properties priced around £{price_input:,.0f}"
            elif filter_type == "Maximum price":
                price_max = price_input
                query = f"Properties under £{price_input:,.0f}"
            else:  # Price range
                if min_price_input >= price_input:
                    st.error("Minimum price must be less than maximum price")
                    st.stop()
                price_min = min_price_input
                price_max = price_input
                query = f"Properties between £{min_price_input:,.0f} and £{price_input:,.0f}"
    
    else:  # Custom Search
        col1, col2 = st.columns(2)
        with col1:
            bedrooms = st.number_input("Minimum bedrooms:", min_value=0, value=0)
            max_price_input = st.number_input("Maximum price:", min_value=0, value=0)
            property_type = st.selectbox(
                "Property type:",
                ["", "Apartment", "Flat", "Terraced", "Detached", "Semi-detached", "Studio"],
                index=0
            )
        with col2:
            bathrooms = st.number_input("Minimum bathrooms:", min_value=0, value=0)
            has_garden = st.checkbox("Has garden")
            min_price_input = st.number_input("Minimum price:", min_value=0, value=0)
        
        filters = []
        if bedrooms > 0:
            filters.append(f"{bedrooms}+ bedrooms")
        if bathrooms > 0:
            filters.append(f"{bathrooms}+ bathrooms")
        if max_price_input > 0:
            price_max = max_price_input
            filters.append(f"under £{max_price_input:,.0f}")
        if min_price_input > 0:
            price_min = min_price_input
            filters.append(f"over £{min_price_input:,.0f}")
        if has_garden:
            filters.append("with garden")
        
        query = "Properties " + ", ".join(filters) if filters else "All properties"
    
    # Clean up property_type
    if property_type == "":
        property_type = None
    
    return {
        'query': query,
        'price_min': price_min,
        'price_max': price_max,
        'property_type': property_type
    }

def search_properties(query, price_min=None, price_max=None, property_type=None):
    """Searches for properties using the backend API with filtering options"""
    with st.spinner('🔍 Searching for your perfect property...'):
        try:
            # Prepare request payload
            payload = {
                "query": query,
                "use_ai_response": True  # Enable AI insights by default
            }
            if price_min is not None:
                payload["price_min"] = price_min
            if price_max is not None:
                payload["price_max"] = price_max
            if property_type:
                payload["property_type"] = property_type
            
            response = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/query",
                        json=payload,
                        timeout=60  # Increased timeout for first search with embeddings
                    )
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if attempt == 2:
                        raise
                    st.warning(f"Retrying connection... (attempt {attempt + 2}/3)")
                    import time
                    time.sleep(1)

            if not response:
                st.error("Failed to connect to the backend")
                return None

            results = response.json()
            
            # Handle the new RAGResponse format
            if "properties" in results:
                if not results["properties"]:
                    st.warning("No properties found matching your criteria.")
                    return None
                
                # Extract documents and metadata from properties
                documents = []
                metadatas = []
                
                for prop in results["properties"]:
                    if isinstance(prop, dict):
                        # Check if it's in the search result format with content/metadata
                        if 'content' in prop and 'metadata' in prop:
                            documents.append(prop['content'])
                            metadatas.append(prop['metadata'])
                        else:
                            # It's a direct property object
                            # Create searchable text as document
                            doc_text = f"Property Type: {prop.get('type', 'N/A')}, "
                            doc_text += f"Price: £{prop.get('price', 0):,.2f}, "
                            doc_text += f"Address: {prop.get('address', 'N/A')}, "
                            doc_text += f"Bedrooms: {prop.get('bedrooms', 0)}, "
                            doc_text += f"Bathrooms: {prop.get('bathrooms', 0)}"
                            
                            documents.append(doc_text)
                            metadatas.append(prop)
                    else:
                        # Fallback for unexpected format
                        documents.append(str(prop))
                        metadatas.append({})
                
                # Convert to old format for compatibility
                converted_results = {
                    "status": "success",
                    "results": {
                        "documents": documents,
                        "metadatas": metadatas
                    },
                    "ai_response": results.get("ai_response"),
                    "query_intent": results.get("query_intent"),
                    "total_found": results.get("total_found", len(results["properties"]))
                }
                return converted_results
            else:
                st.error("Failed to get results from the backend")
                return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

def create_property_map(properties_data):
    """Creates an interactive map with property locations"""
    try:
        # Filter properties with valid coordinates
        valid_properties = [p for p in properties_data 
                          if p.get('latitude') is not None and p.get('longitude') is not None]
        
        if not valid_properties:
            st.warning("No properties with valid location data to display on map")
            return
        
        st.markdown("""
            <div class="property-card">
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🗺️ Property Locations</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate center of the map
        center_lat = sum(p['latitude'] for p in valid_properties) / len(valid_properties)
        center_lon = sum(p['longitude'] for p in valid_properties) / len(valid_properties)
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each property
        for prop in valid_properties:
            try:
                # Create popup content
                popup_content = f"""
                <div style='width: 200px'>
                    <h4>{prop['type']}</h4>
                    <p><strong>Price:</strong> £{prop['price']:,.2f}</p>
                    <p><strong>Bedrooms:</strong> {prop['bedrooms']}</p>
                    <p><strong>Bathrooms:</strong> {prop['bathrooms']}</p>
                    <p><strong>Address:</strong> {prop['address']}</p>
                </div>
                """
                
                # Add marker with popup
                folium.Marker(
                    location=[prop['latitude'], prop['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(marker_cluster)
            except Exception as e:
                st.warning(f"Could not add marker for property: {str(e)}")
                continue
        
        # Display the map
        st.markdown("""
            <div class="property-card">
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🗺️ Property Locations</h3>
            </div>
        """, unsafe_allow_html=True)
        
        folium_static(m)
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return

def get_property_ai_insights(meta):
    """Generate AI insights for a single property with optimized context"""
    try:
        # Create property-specific context
        property_type = meta.get('type', 'property')
        price = meta.get('price', 0)
        bedrooms = meta.get('bedrooms', 0)
        bathrooms = meta.get('bathrooms', 0)
        location = meta.get('address', 'unknown location')
        crime_score = meta.get('crime_score_weight') or meta.get('crime_score')
        flood_risk = meta.get('flood_risk', 'Unknown')
        
        # Create a more specific and focused query for individual property analysis
        query = f"""Provide specific insights and recommendations for this property:
        - Type: {property_type}
        - Price: £{price:,.2f}
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Location: {location}
        - Crime Score: {crime_score if crime_score else 'Not available'}
        - Flood Risk: {flood_risk}
        
        Focus on: investment potential, suitability for different buyers, location benefits, value assessment, and any specific recommendations."""
        
        # Use the API for consistent results
        payload = {
            "query": query,
            "use_ai_response": True,
            "max_results": 1
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            results = response.json()
            if results.get("ai_response"):
                return results["ai_response"]
        
        return None
        
    except Exception as e:
        print(f"Error getting property AI insights: {e}")
        return None

def display_property_card(doc, meta):
    """Displays a single property card with safe key access"""
    # Safely extract metadata with defaults
    property_type = meta.get('type', 'Property') if meta else 'Property'
    price = meta.get('price', 0) if meta else 0
    bedrooms = meta.get('bedrooms', 0) if meta else 0
    bathrooms = meta.get('bathrooms', 0) if meta else 0
    address = meta.get('address', 'Address not available') if meta else 'Address not available'
    
    listing_date = 'N/A'
    if meta and 'listing_date' in meta and meta['listing_date']:
        try:
            if isinstance(meta['listing_date'], str):
                listing_date = pd.to_datetime(meta['listing_date']).strftime('%Y-%m-%d')
            elif isinstance(meta['listing_date'], (pd.Timestamp, np.datetime64)):
                listing_date = pd.Timestamp(meta['listing_date']).strftime('%Y-%m-%d')
            else:
                st.warning(f"Unexpected date format: {type(meta['listing_date'])}")
        except Exception as e:
            st.warning(f"Error processing date: {str(e)}")
            pass
    
    flood_risk = meta.get('flood_risk', 'N/A') if meta else 'N/A'
    flood_color = "#28a745" if flood_risk == 'None' else "#ffc107" if flood_risk == 'Low' else "#dc3545"
    
    st.markdown(f"""
        <div class="property-card">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='color: #2c3e50; margin-bottom: 0.5rem;'>{property_type}</h3>
                    <p style='color: #6c757d; margin: 0;'>Listed: {listing_date}</p>
                </div>
                <div class="price-tag">£{price:,.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Key features with icons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>🛏️ Bedrooms</h4>
                <p style='font-size: 1.5rem; color: #2c3e50;'>{bedrooms}</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>🚿 Bathrooms</h4>
                <p style='font-size: 1.5rem; color: #2c3e50;'>{bathrooms}</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        crime_score = None
        if meta and "crime_score_weight" in meta and meta["crime_score_weight"] is not None:
            try:
                crime_score = float(meta["crime_score_weight"])
            except (ValueError, TypeError):
                crime_score = None
        elif meta and "crime_score" in meta and meta["crime_score"] is not None:
            try:
                crime_score = float(meta["crime_score"])
            except (ValueError, TypeError):
                crime_score = None
            
        if crime_score is not None:
            crime_color = "#28a745" if crime_score < 0.4 else "#ffc107" if crime_score < 0.7 else "#dc3545"
            st.markdown(f"""
                <div class="metric-tile">
                    <h4 style='color: #6c757d;'>🔒 Crime Score</h4>
                    <p style='font-size: 1.5rem; color: {crime_color};'>{crime_score:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-tile">
                    <h4 style='color: #6c757d;'>🔒 Crime Score</h4>
                    <p style='font-size: 1.5rem; color: #6c757d;'>N/A</p>
                </div>
            """, unsafe_allow_html=True)
    with col4:
        flood_risk = meta.get('flood_risk', 'N/A')
        flood_color = "#28a745" if flood_risk == 'None' else "#ffc107" if flood_risk == 'Low' else "#dc3545"
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>💧 Flood Risk</h4>
                <p style='font-size: 1.5rem; color: {flood_color};'>{flood_risk}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Location with map-style card
    st.markdown(f"""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #2c3e50;'>📍 Location</h4>
            <p style='color: #6c757d; margin: 0;'>{address}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Property description
    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #2c3e50;'>📝 Property Description</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Format the description properly
    if isinstance(doc, dict):
        # If doc is a dictionary, extract meaningful information
        description_parts = []
        if 'property_type_full_description' in doc and doc['property_type_full_description']:
            description_parts.append(str(doc['property_type_full_description']))
        if 'type' in doc and doc['type']:
            description_parts.append(f"Property Type: {doc['type']}")
        if 'bedrooms' in doc and doc['bedrooms']:
            description_parts.append(f"Bedrooms: {doc['bedrooms']}")
        if 'bathrooms' in doc and doc['bathrooms']:
            description_parts.append(f"Bathrooms: {doc['bathrooms']}")
        if 'address' in doc and doc['address']:
            description_parts.append(f"Location: {doc['address']}")
        
        # Filter out None values and ensure all items are strings
        description_parts = [str(part) for part in description_parts if part is not None and str(part).strip()]
        description = ". ".join(description_parts) if description_parts else "Property details available in the information above."
    elif isinstance(doc, str):
        # If doc is a string, clean it up
        description = re.sub(r'<[^>]+>', '', doc)
        description = description.replace('Property Type:', '')\
                              .replace('Price:', '')\
                              .replace('Bedrooms:', '')\
                              .replace('Bathrooms:', '')\
                              .replace('Location:', '')\
                              .strip()
        if not description:
            description = "Property details available in the information above."
    else:
        # Create a basic description from metadata
        description = f"This is a {meta.get('type', 'property')} with {meta.get('bedrooms', 'N/A')} bedrooms and {meta.get('bathrooms', 'N/A')} bathrooms, located in {meta.get('address', 'an undisclosed location')}."
    
    st.markdown(f"""
        <div style='padding: 0 1rem;'>
            <p style='color: #6c757d; line-height: 1.6;'>{description}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add AI Recommendations section
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: white; margin-bottom: 1rem;'>🤖 AI Property Insights</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if we have AI insights in session state for this property
    property_key = f"ai_insights_{hash(str(meta))}"
    
    if property_key in st.session_state:
        # Display cached AI insights
        ai_insights = st.session_state[property_key]
        st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                <div style='color: #2c3e50; line-height: 1.6;'>{ai_insights}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Show generate button
        if st.button(f"💡 Generate AI Insights", key=f"insights_{hash(str(meta))}"):
            with st.spinner("🧠 Analyzing this property..."):
                ai_insights = get_property_ai_insights(meta)
                if ai_insights:
                    st.session_state[property_key] = ai_insights
                    st.markdown(f"""
                        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                            <div style='color: #2c3e50; line-height: 1.6;'>{ai_insights}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ AI insights generated!")
                    st.rerun()
                else:
                    st.error("❌ Could not generate AI insights for this property.")
        
        # Show placeholder message
        st.markdown("""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                <p style='color: #6c757d; margin: 0; text-align: center;'>
                    💡 Click the button above to get AI-powered insights about this specific property
                </p>
            </div>
        """, unsafe_allow_html=True)

def display_analytics(df):
    """Displays analytics for the search results"""
    # Check if DataFrame is empty
    if df is None or df.empty:
        st.warning("No data available for analytics")
        return
    
    matched_properties = len(df)
    
    # Quick stats with enhanced styling
    st.markdown("""
        <div class="property-card">
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📈 Quick Statistics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = df['price'].mean()
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>Average Price</h4>
                <p style='font-size: 1.5rem; color: #2c3e50;'>£{avg_price:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        price_range = f"£{df['price'].min():,.0f} - £{df['price'].max():,.0f}"
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>Price Range</h4>
                <p style='font-size: 1.5rem; color: #2c3e50;'>{price_range}</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_beds = df['bedrooms'].mean()
        st.markdown(f"""
            <div class="metric-tile">
                <h4 style='color: #6c757d;'>Avg Bedrooms</h4>
                <p style='font-size: 1.5rem; color: #2c3e50;'>{avg_beds:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Visual insights
    if matched_properties > 1:
        st.markdown("""
            <div class="property-card" style='margin-top: 2rem;'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📊 Visual Insights</h3>
            </div>
        """, unsafe_allow_html=True)
        
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            # Price Distribution
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.hist(df['price'], bins=min(10, matched_properties), edgecolor='black', color='#FF4B4B')
            ax.set_title('Price Distribution')
            ax.set_xlabel('Price (£)')
            ax.set_ylabel('Number of Properties')
            st.pyplot(fig)
        
        with viz_col2:
            # Property Types
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            type_counts = df['type'].value_counts()
            ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                  colors=plt.cm.Pastel1(np.linspace(0, 1, len(type_counts))))
            ax.set_title('Property Types')
            st.pyplot(fig)
        
        # Correlation Analysis
        if matched_properties >= 5:
            # Pattern Analysis
            st.markdown("""
                <div class="property-card" style='margin-top: 2rem;'>
                    <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🔍 Pattern Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = ['price', 'bedrooms', 'bathrooms']
            if 'crime_score' in df.columns:
                numeric_cols.append('crime_score')
            
            corr = df[numeric_cols].corr()
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlations')
            st.pyplot(fig)
            
            # Price Trends
            st.markdown("""
                <div class="property-card" style='margin-top: 2rem;'>
                    <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📈 Price Trends</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Handle listing date
            if 'listing_date' in df.columns:
                df['listing_date'] = pd.to_datetime(df['listing_date'])
                
                # Group by listing date and calculate average price
                daily_prices = df.groupby('listing_date')['price'].mean().reset_index()
                daily_prices = daily_prices.sort_values('listing_date')
            else:
                st.warning("Listing date information is not available for price trends.")
            
            # Create price trend visualization
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.plot(daily_prices['listing_date'], daily_prices['price'], 
                   marker='o', linestyle='-', color='#FF4B4B')
            ax.set_title('Average Property Prices Over Time')
            ax.set_xlabel('Listing Date')
            ax.set_ylabel('Average Price (£)')
            fig.autofmt_xdate()  # Rotate x-axis labels
            st.pyplot(fig)
            
            # Area Statistics
            st.markdown("""
                <div class="property-card" style='margin-top: 2rem;'>
                    <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📍 Area Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Group properties by area and calculate statistics
            # Check if address column exists and has data
            if 'address' not in df.columns or df['address'].isnull().all():
                st.warning("No address data available for area analysis")
            else:
                # Prepare aggregation dictionary
                agg_dict = {
                    'price': ['mean', 'min', 'max', 'count']
                }
                
                # Add crime score if available
                if 'crime_score_weight' in df.columns:
                    agg_dict['crime_score_weight'] = 'mean'
                elif 'crime_score' in df.columns:
                    agg_dict['crime_score'] = 'mean'
                    
                # Add flood risk if available
                if 'flood_risk' in df.columns:
                    agg_dict['flood_risk'] = lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'Unknown'
                    
                try:
                    area_stats = df.groupby('address').agg(agg_dict).reset_index()
                    
                    # Check if groupby result is empty
                    if area_stats.empty:
                        st.warning("No area statistics available")
                    else:
                        # Continue with the analytics display
                        pass  # Rest of the analytics code would go here
                except Exception as e:
                    st.warning(f"Unable to generate area statistics: {str(e)}")
            
            # Rename columns
            # Rename columns
            new_columns = {
                ('address', ''): 'Area',
                ('price', 'mean'): 'Avg Price',
                ('price', 'min'): 'Min Price',
                ('price', 'max'): 'Max Price',
                ('price', 'count'): 'Properties'
            }
            
            if ('crime_score_weight', 'mean') in area_stats.columns:
                new_columns[('crime_score_weight', 'mean')] = 'Crime Score'
            elif ('crime_score', 'mean') in area_stats.columns:
                new_columns[('crime_score', 'mean')] = 'Crime Score'
                
            if ('flood_risk', '<lambda_0>') in area_stats.columns:
                new_columns[('flood_risk', '<lambda_0>')] = 'Typical Flood Risk'
            
            area_stats.columns = area_stats.columns.map(lambda x: new_columns.get(x, x))
            
            # Format the display
            format_dict = {
                'Avg Price': '£{:,.2f}',
                'Min Price': '£{:,.2f}',
                'Max Price': '£{:,.2f}'
            }
            
            if 'Crime Score' in area_stats.columns:
                format_dict['Crime Score'] = '{:.2f}'
            
            # Display area statistics
            st.dataframe(
                area_stats.style.format(format_dict),
                use_container_width=True
            )
    
    # Key insights
    st.markdown("""
        <div class="property-card" style='margin-top: 2rem;'>
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🎯 Key Insights</h3>
        </div>
    """, unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    with insights_col1:
        price_per_bedroom = df['price'] / df['bedrooms']
        st.markdown(f"""
            <div class="insight-card">
                <h4 style='color: #2c3e50;'>💰 Price Insights</h4>
                <p>• Average price per bedroom: £{price_per_bedroom.mean():,.2f}</p>
                <p>• Most common property type: {df['type'].mode().iloc[0]}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        if 'crime_score' in df.columns:
            avg_crime = df['crime_score'].mean()
            st.markdown(f"""
                <div class="insight-card">
                    <h4 style='color: #2c3e50;'>📍 Location Analysis</h4>
                    <p>• Average crime score: {avg_crime:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

def display_guide():
    """Displays the guide tab content in a structured and visually appealing way"""
    # Header
    st.title("ℹ️ User Guide")
    st.write("How to make the most of your property search")
    
    # Search Methods
    st.header("🔍 Search Methods")
    
    # Method 1: Natural Questions
    st.subheader("1. Natural Questions")
    st.write("Ask anything about properties in natural language")
    st.info('Example: "Show me 3-bedroom houses under £2000"')
    
    # Method 2: Price Filter
    st.subheader("2. Price Filter")
    st.write("Search by specific price ranges")
    st.info("Example: Set exact price or range with optional margin")
    
    # Method 3: Custom Search
    st.subheader("3. Custom Search")
    st.write("Use multiple filters for precise results")
    st.info("Example: Combine bedrooms, bathrooms, and other features")
    
    # Tips & Tricks
    st.header("💡 Tips & Tricks")
    tips = [
        "Use natural language for more flexible searches",
        "Combine multiple criteria for better results",
        "Check the Analytics tab for insights",
        "Save interesting properties for later"
    ]
    for tip in tips:
        st.markdown(f"• {tip}")

def fetch_property_stats():
    """Fetches overall property statistics from the backend"""
    try:
        stats = None
        for attempt in range(3):
            try:
                stats = requests.get(
                    "http://127.0.0.1:8000/stats",
                    timeout=30  # Increased timeout
                ).json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == 2:
                    raise
                st.sidebar.warning(f"Retrying connection... (attempt {attempt + 2}/3)")
                import time
                time.sleep(1)

        if stats:
            st.sidebar.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                    <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📊 Market Overview</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown(f"""
                <div class="metric-tile" style='margin-bottom: 1rem;'>
                    <h4 style='color: #6c757d;'>Total Properties</h4>
                    <p style='font-size: 1.5rem; color: #2c3e50;'>{stats['total_properties']}</p>
                </div>
                
                <div class="metric-tile" style='margin-bottom: 1rem;'>
                    <h4 style='color: #6c757d;'>Average Price</h4>
                    <p style='font-size: 1.5rem; color: #2c3e50;'>£{stats['avg_price']:,.2f}</p>
                </div>
                
                <div class="metric-tile" style='margin-bottom: 1rem;'>
                    <h4 style='color: #6c757d;'>Average Bedrooms</h4>
                    <p style='font-size: 1.5rem; color: #2c3e50;'>{stats['avg_bedrooms']:.1f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;'>
                    <h4 style='color: #2c3e50;'>Property Types</h4>
                </div>
            """, unsafe_allow_html=True)
            
            prop_types_df = pd.DataFrame(
                list(stats["property_types"].items()),
                columns=["Type", "Count"]
            )
            st.sidebar.dataframe(
                prop_types_df,
                hide_index=True,
                use_container_width=True
            )
    except Exception as e:
        st.sidebar.error("Could not load market statistics")

# Main application logic
with tab1:
    st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
            <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Find Your Dream Property</h2>
        </div>
    """, unsafe_allow_html=True)
    
    query_params = create_search_form()
    
    if query_params and query_params['query']:
        # Store the query in session state for AI insights
        st.session_state.last_query = query_params['query']
        
        results = search_properties(
            query_params['query'], 
            price_min=query_params['price_min'],
            price_max=query_params['price_max'],
            property_type=query_params['property_type']
        )
        if results and results.get("results") and results["results"].get("documents"):
            matched_properties = len(results["results"]["documents"])
            
            # Check if we actually have properties
            if matched_properties == 0:
                st.warning("No properties found matching your criteria.")
                st.session_state.last_search_results = []
            else:
                # Store results for analytics
                properties_data = []
                for doc, meta in zip(results["results"]["documents"], results["results"]["metadatas"]):
                    # Create basic property data with null safety
                    property_data = {
                        'price': float(meta['price']) if meta.get('price') is not None else 0.0,
                        'bedrooms': int(meta['bedrooms']) if meta.get('bedrooms') is not None else 0,
                        'bathrooms': int(meta['bathrooms']) if meta.get('bathrooms') is not None else 0,
                        'type': meta.get('type', 'Unknown'),
                        'address': meta.get('address', 'Address not available'),
                        'description': doc,
                        'crime_score': None,  # Initialize with default values
                        'listing_date': None,
                        'flood_risk': 'N/A',
                        'latitude': None,
                        'longitude': None
                    }
                    
                    # Update with available data
                    try:
                        # Handle crime score
                        if 'crime_score_weight' in meta and meta['crime_score_weight'] is not None:
                            try:
                                property_data['crime_score'] = float(meta['crime_score_weight'])
                            except (ValueError, TypeError):
                                property_data['crime_score'] = None
                        elif 'crime_score' in meta and meta['crime_score'] is not None:
                            try:
                                property_data['crime_score'] = float(meta['crime_score'])
                            except (ValueError, TypeError):
                                property_data['crime_score'] = None
                        
                        # Handle listing date
                        if 'listing_date' in meta and meta['listing_date']:
                            try:
                                if isinstance(meta['listing_date'], str):
                                    property_data['listing_date'] = pd.to_datetime(meta['listing_date'])
                                elif isinstance(meta['listing_date'], (pd.Timestamp, np.datetime64)):
                                    property_data['listing_date'] = pd.Timestamp(meta['listing_date'])
                                else:
                                    st.warning(f"Unexpected date format in data: {type(meta['listing_date'])}")
                            except Exception as e:
                                st.warning(f"Error processing date in data: {str(e)}")
                                property_data['listing_date'] = None
                        
                        # Handle flood risk
                        if 'flood_risk' in meta:
                            property_data['flood_risk'] = meta['flood_risk']
                        
                        # Handle coordinates
                        if 'latitude' in meta and 'longitude' in meta:
                            if meta['latitude'] and meta['longitude']:
                                property_data['latitude'] = float(meta['latitude'])
                                property_data['longitude'] = float(meta['longitude'])
                    except Exception as e:
                        st.error(f"Error processing property data: {str(e)}")
                        continue  # Skip this property if there's an error
                    
                    properties_data.append(property_data)
                st.session_state.last_search_results = properties_data
                
                # Show results count and sorting options
                st.markdown(f"""
                    <div class="property-card" style='text-align: center;'>
                        <h2 style='color: #2c3e50;'>🎯 Found {matched_properties} Properties</h2>
                        <p style='color: #6c757d;'>Sort and explore your matches</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Sorting options
                col1, col2 = st.columns([2, 1])
                with col1:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["Price (Low to High)", "Price (High to Low)", 
                         "Bedrooms", "Bathrooms", "Crime Score (Low to High)",
                         "Listing Date (Newest)", "Listing Date (Oldest)"]
                    )
            
            # Sort properties based on selection
            if sort_by == "Price (Low to High)":
                properties_data.sort(key=lambda x: x['price'])
            elif sort_by == "Price (High to Low)":
                properties_data.sort(key=lambda x: x['price'], reverse=True)
            elif sort_by == "Bedrooms":
                properties_data.sort(key=lambda x: x['bedrooms'], reverse=True)
            elif sort_by == "Bathrooms":
                properties_data.sort(key=lambda x: x['bathrooms'], reverse=True)
            elif sort_by == "Crime Score (Low to High)":
                properties_data.sort(key=lambda x: x['crime_score'])
            elif sort_by == "Listing Date (Newest)":
                properties_data.sort(key=lambda x: pd.to_datetime(x['listing_date']), reverse=True)
            elif sort_by == "Listing Date (Oldest)":
                properties_data.sort(key=lambda x: pd.to_datetime(x['listing_date']))
                
            # Add export button
            with col2:
                if st.button("📥 Export Results"):
                    df_export = pd.DataFrame(properties_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="property_search_results.csv",
                        mime="text/csv"
                    )
            
            # Display map view first
            create_property_map(properties_data)
            
            # AI Insights Control Panel
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                    <h3 style='color: white; margin-bottom: 1rem;'>🤖 AI Property Insights Control</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                    <p style='color: #6c757d; margin: 0;'>
                        Get personalized AI insights for each property or generate insights for all properties at once.
                    </p>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🚀 Generate All AI Insights", key="bulk_insights", help="Generate AI insights for all properties in results"):
                    progress_bar = st.progress(0)
                    total_properties = len(results["results"]["metadatas"])
                    
                    with st.spinner("🧠 Generating AI insights for all properties..."):
                        for i, meta in enumerate(results["results"]["metadatas"]):
                            property_key = f"ai_insights_{hash(str(meta))}"
                            if property_key not in st.session_state:
                                # Generate insights for this property
                                ai_insights = get_property_ai_insights(meta)
                                if ai_insights:
                                    st.session_state[property_key] = ai_insights
                            
                            # Update progress
                            progress_bar.progress((i + 1) / total_properties)
                    
                    st.success(f"✅ Generated AI insights for {total_properties} properties!")
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display property cards
            for i, (doc, meta) in enumerate(zip(
                results["results"]["documents"],
                results["results"]["metadatas"]
            )):
                # Safely get property type with fallback
                property_type = meta.get('type', 'Property') if meta else 'Property'
                with st.expander(f"🏠 Property {i+1} - {property_type}", expanded=i==0):
                    display_property_card(doc, meta)
            
            # Display AI Property Insights
            if results.get("ai_response"):
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                        <h3 style='color: white; margin-bottom: 1rem;'>🤖 AI Property Insights</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(f"""
                        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                            <div style='color: #2c3e50; line-height: 1.6;'>{results["ai_response"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Show loading message or request AI insights
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                        <h3 style='color: white; margin-bottom: 1rem;'>🤖 AI Property Insights</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("💡 Generate AI Insights", key="generate_insights"):
                    with st.spinner("🧠 Analyzing properties and generating insights..."):
                        try:
                            # Make a new request with AI response enabled
                            ai_payload = {
                                "query": st.session_state.get('last_query', ''),
                                "use_ai_response": True,
                                "max_results": 5
                            }
                            
                            ai_response = requests.post(
                                "http://127.0.0.1:8000/query",
                                json=ai_payload,
                                timeout=30
                            )
                            
                            if ai_response.status_code == 200:
                                ai_results = ai_response.json()
                                if ai_results.get("ai_response"):
                                    st.markdown(f"""
                                        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                                            <div style='color: #2c3e50; line-height: 1.6;'>{ai_results["ai_response"]}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    st.success("✅ AI insights generated successfully!")
                                else:
                                    st.warning("⚠️ AI insights could not be generated at this time.")
                            else:
                                st.error("❌ Failed to generate AI insights. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {str(e)}")
                
                st.markdown("""
                    <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color: #6c757d; margin: 0; text-align: center;'>
                            💡 Click the button above to get AI-powered insights about your search results
                        </p>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class="gradient-header" style='margin-bottom: 2rem;'>
            <h2 style='margin: 0;'>📊 Property Analytics Dashboard</h2>
            <p style='opacity: 0.9;'>Insights from your property search</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_search_results:
        df = pd.DataFrame(st.session_state.last_search_results)
        display_analytics(df)
    else:
        st.info("👆 Please use the Search tab to find properties first!")

with tab3:
    display_guide()

# Example questions in sidebar
st.sidebar.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h3 style='color: #2c3e50;'>🤔 Example Questions</h3>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h4 style='color: #2c3e50;'>💬 Natural Questions</h4>
        <ul style='color: #6c757d;'>
            <li>What's the average price of 3-bedroom homes?</li>
            <li>Show me properties with gardens</li>
            <li>Find houses with low crime scores</li>
        </ul>
    </div>
    
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h4 style='color: #2c3e50;'>💰 Price Queries</h4>
        <ul style='color: #6c757d;'>
            <li>Properties under £2000</li>
            <li>Homes between £1500-£3000</li>
            <li>Most expensive properties</li>
        </ul>
    </div>
    
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
        <h4 style='color: #2c3e50;'>🏠 Specific Features</h4>
        <ul style='color: #6c757d;'>
            <li>2-bedroom properties with parking</li>
            <li>Houses with garden and 2+ bathrooms</li>
            <li>Apartments with good safety scores</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Display overall statistics
fetch_property_stats()