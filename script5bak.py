import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
import os
import tempfile
import json
import re
import io
import base64
import requests
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union, Any
from sqlalchemy import create_engine, text 
from scipy.ndimage import distance_transform_edt
import folium
from folium.raster_layers import ImageOverlay
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from geopy.geocoders import Nominatim
import google.generativeai as genai
from google.generativeai import GenerativeModel
from shapely.geometry import shape, box, Point
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import PIL.Image
from pylusat import distance, rescale, utils, geotools

# Set page config for initial load
st.set_page_config(
    page_title="Advanced Land Suitability Analysis", 
    page_icon="🏙️", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/land-suitability-analysis',
        'Report a bug': 'https://github.com/yourusername/land-suitability-analysis/issues',
        'About': "# Land Suitability Analysis Tool\nThis application helps urban planners identify optimal locations based on multiple criteria."
    }
)

# Define color scheme for the app
COLOR_PRIMARY = "#4A90E2"   # Primary blue
COLOR_SECONDARY = "#29B6F6" # Light blue
COLOR_SUCCESS = "#66BB6A"   # Green
COLOR_WARNING = "#FFA726"   # Orange
COLOR_DANGER = "#EF5350"    # Red
COLOR_BACKGROUND = "#F5F7F9" # Light grayish blue
COLOR_TEXT = "#2C3E50"      # Dark blue-gray

# Custom CSS to improve styling
def load_custom_css():
    st.markdown('''
    <style>
    .main {
        background-color: #F5F7F9;
        color: #2C3E50;
    }
    .stButton > button {
        color: white;
        background-color: #4A90E2;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2979B5;
    }
    .stProgress > div > div {
        background-color: #4A90E2;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stAlert > div {
        border-radius: 4px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    .stSidebar .sidebar-content {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border-right: 1px solid #E0E0E0;
        border-left: 1px solid #E0E0E0;
        border-top: 1px solid #E0E0E0;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .dataframe {
        font-family: 'Source Sans Pro', sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #4A90E2;
        color: white;
        text-align: left;
        padding: 12px;
    }
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #E0E0E0;
    }
    .dataframe tr:nth-child(even) {
        background-color: #F8F9FA;
    }
    </style>
    ''', unsafe_allow_html=True)

#############################################
# Data Models - Classes to structure our data
#############################################

class DatabaseConfig:
    """Database connection configuration."""
    def __init__(self, host: str, port: str, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

class AnalysisGrid:
    """Represents the spatial grid for analysis."""
    def __init__(self, transform, width: int, height: int, utm_crs: str, bounds: Tuple):
        self.transform = transform
        self.width = width
        self.height = height
        self.utm_crs = utm_crs
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)
    
    def get_extent(self) -> List[float]:
        """Get extent in format suitable for visualization."""
        xmin, ymin, xmax, ymax = self.bounds
        return [xmin, xmax, ymin, ymax]

class AnalysisCriteria:
    """Stores criteria for land suitability analysis."""
    def __init__(self, 
                 layers: List[str], 
                 distance_requirements: Dict[str, float], 
                 weights: Dict[str, int], 
                 avoid: List[str], 
                 objective: str):
        self.layers = layers
        self.distance_requirements = distance_requirements
        self.weights = weights
        self.avoid = avoid
        self.objective = objective
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "layers": self.layers,
            "distance_requirements": self.distance_requirements,
            "weights": self.weights,
            "avoid": self.avoid,
            "objective": self.objective
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisCriteria':
        """Create AnalysisCriteria from dictionary."""
        return cls(
            layers=data.get("layers", []),
            distance_requirements=data.get("distance_requirements", {}),
            weights=data.get("weights", {}),
            avoid=data.get("avoid", []),
            objective=data.get("objective", "")
        )

class SuitableLocation:
    """Represents a suitable location found by analysis."""
    def __init__(self, 
                 location_id: int,
                 geometry: Point,
                 area_hectares: float, 
                 suitability_score: float,
                 latitude: float,
                 longitude: float,
                 address: Optional[str] = None,
                 neighborhood: Optional[str] = None,
                 city: Optional[str] = None,
                 nearby_features: Optional[Dict] = None):
        self.location_id = location_id
        self.geometry = geometry
        self.area_hectares = area_hectares
        self.suitability_score = suitability_score
        self.latitude = latitude
        self.longitude = longitude
        self.address = address
        self.neighborhood = neighborhood
        self.city = city
        self.nearby_features = nearby_features or {}
        self.explanation = ""
        self.considerations = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.location_id,
            "area_hectares": float(self.area_hectares),
            "suitability_score": float(self.suitability_score),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "address": str(self.address) if self.address else None,
            "neighborhood": str(self.neighborhood) if self.neighborhood else None,
            "city": str(self.city) if self.city else None
        }

class AnalysisResult:
    """Stores the results of a land suitability analysis."""
    def __init__(self, 
                 suitability_raster: np.ndarray,
                 grid: AnalysisGrid,
                 locations: List[SuitableLocation],
                 criteria: AnalysisCriteria,
                 timestamp: datetime = None,
                 gemini_analysis: Optional[Dict] = None):
        self.suitability_raster = suitability_raster
        self.grid = grid
        self.locations = locations
        self.criteria = criteria
        self.timestamp = timestamp or datetime.now()
        self.gemini_analysis = gemini_analysis or {}
    
    def get_top_locations(self, n: int = 3) -> List[SuitableLocation]:
        """Get top n locations by suitability score."""
        return sorted(self.locations, key=lambda x: x.suitability_score, reverse=True)[:n]
        
    def to_json_serializable(self):
        """Convert to JSON serializable format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'criteria': self.criteria.to_dict(),
            'locations_count': len(self.locations),
            'grid_info': {
                'width': self.grid.width,
                'height': self.grid.height,
                'utm_crs': str(self.grid.utm_crs),
                'bounds': list(self.grid.bounds)
            },
            'gemini_analysis': self.gemini_analysis
        }

#############################################
# Database & Remote API Services 
#############################################

class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize with database configuration."""
        self.config = config
        self.engine = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            conn_string = self.config.get_connection_string()
            self.engine = create_engine(conn_string)
            
            # Test the connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
        except Exception as e:
            st.error(f"Database connection error: {e}")
            st.warning("Please check your database credentials and ensure PostgreSQL server is running.")
            return False
    
    def get_available_layers(self) -> List[str]:
        """Get list of available tables in the database."""
        if not self.engine:
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
                return [row[0] for row in result]
        except Exception as e:
            st.error(f"Error checking available tables: {e}")
            return []
    
    def fetch_layer(self, layer_name: str) -> Optional[gpd.GeoDataFrame]:
        """Fetch a specific layer from the database."""
        if not self.engine:
            st.error("No database connection available.")
            return None
        
        try:
            query = f"SELECT * FROM {layer_name}"
            gdf = gpd.read_postgis(query, self.engine, geom_col='geom')
            
            # Check if the layer has any rows
            if len(gdf) == 0:
                st.warning(f"Layer {layer_name} exists but contains no data.")
                return None
            
            # Ensure the geometry column is properly set
            if 'geom' in gdf.columns:
                gdf = gdf.set_geometry('geom')
            
            # Check CRS
            if gdf.crs is None:
                st.warning(f"Layer {layer_name} has no CRS information. Assuming WGS84 (EPSG:4326)")
                gdf.set_crs(epsg=4326, inplace=True)
            
            return gdf
        except Exception as e:
            st.error(f"Could not fetch layer {layer_name}: {e}")
            return None
    
    def fetch_layers(self, layer_names: List[str]) -> Dict[str, gpd.GeoDataFrame]:
        """Fetch multiple layers from the database."""
        layers = {}
        available_tables = self.get_available_layers()
        
        for layer_name in layer_names:
            if layer_name not in available_tables:
                st.warning(f"Layer {layer_name} does not exist in the database.")
                continue
            
            gdf = self.fetch_layer(layer_name)
            if gdf is not None:
                layers[layer_name] = gdf
                st.success(f"Successfully loaded {layer_name} with {len(gdf)} features.")
        
        return layers
    
    def get_nearby_features(self, locations: List[SuitableLocation], radius_meters: float = 1000) -> Dict[str, Dict]:
        """Find nearby features from database layers for locations."""
        if not self.engine:
            return {}
        
        result = {}
        available_layers = self.get_available_layers()
        
        # Layers to check (exclude any that aren't in the database)
        layers_to_check = ['schools', 'healthcare', 'bank', 'roads', 'railway', 'masjids']
        valid_layers = [layer for layer in layers_to_check if layer in available_layers]
        
        for location in locations:
            location_id = str(location.location_id)
            lat = location.latitude
            lon = location.longitude
            
            location_features = {}
            
            for layer in valid_layers:
                try:
                    # SQL query to find features within radius
                    query = f"""
                    SELECT name, ST_Distance(
                        ST_Transform(geom, 3857),
                        ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857)
                    ) as distance
                    FROM {layer}
                    WHERE ST_DWithin(
                        ST_Transform(geom, 3857),
                        ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857),
                        {radius_meters}
                    )
                    ORDER BY distance
                    LIMIT 5;
                    """
                    
                    # Execute query
                    with self.engine.connect() as conn:
                        query_result = conn.execute(text(query))
                        features = [{"name": row[0], "distance": row[1]} for row in query_result]
                    
                    location_features[layer] = features
                except Exception as e:
                    st.warning(f"Error querying {layer} for location {location_id}: {e}")
            
            result[location_id] = location_features
        
        return result
    
    def load_boundary(self, boundary_name: str = "islamabad", target_crs: str = "EPSG:4326") -> Optional[gpd.GeoDataFrame]:
        """Load a boundary from the database."""
        if not self.engine:
            return None
            
        try:
            gdf = self.fetch_layer(boundary_name)
            
            if gdf is None:
                return None
                
            # Project to the target CRS if needed
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
                
            return gdf
        except Exception as e:
            st.warning(f"Could not load boundary {boundary_name}: {e}")
            return None


class GeminiService:
    """Service for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.model = None
    
    def connect(self) -> bool:
        """Setup Gemini API connection."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            return True
        except Exception as e:
            st.error(f"Error setting up Gemini API: {e}")
            return False
    
    def analyze_criteria(self, user_input: str) -> Optional[AnalysisCriteria]:
        """Analyze user input to extract suitability criteria."""
        if not self.model:
            st.error("Gemini API not configured.")
            return None
        
        prompt = f'''
        Analyze the following user criteria for site suitability analysis:
        "{user_input}"
        
        Using ONLY the following available GIS layers:
        - masjids
        - railway
        - roads
        - healthcare
        - bank
        - schools
        - builtup
        - water
        - bareland
        - bus_stops
        - crops
        Pay special attention to any terms indicating avoidance such as:
        - "avoid"
         - "away from"
         - "not near"
         - "far from"
         - "excluding"
         - "outside of"
         - "not in"
         - "stay clear of"
         - "not on"
         - at least X metres from etc
    
    For builtup areas specifically, if the user indicates any preference to avoid built-up areas, populated areas, 
    developed areas, or urban areas, make sure to include "builtup" in the avoid list.
        
        Identify and return a JSON structure with:
        1. The specific layers needed from the list above. Check for similar keywords. For example, call schools layer, if it says educational institutes, call school etc
        2. Distance requirements for each layer (e.g., if 10m you output 10, if 1km you output 1000 etc)
        3. Weights for each criteria (on a scale of 1-100)
        4. If there are any layers that should be avoided (e.g., away from railways)
        5. The overall objective of the analysis
        
        Format your response as valid JSON without markdown or code formatting.
        Just return the raw JSON object like this:
        {{"layers": [], "distance_requirements": {{}}, "weights": {{}}, "avoid": [], "objective": ""}}
        '''

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.replace("```", "", 1)
                
            response_text = response_text.strip()
            
            # Parse the JSON
            criteria_dict = json.loads(response_text)
            
            # Create and return an AnalysisCriteria object
            return AnalysisCriteria.from_dict(criteria_dict)
        except Exception as e:
            st.error(f"Error analyzing criteria with Gemini: {e}")
            return None
    
    def analyze_locations(self, 
                         locations: List[SuitableLocation], 
                         criteria: AnalysisCriteria, 
                         analysis_summary: str) -> Optional[Dict]:
        """Analyze suitable locations and provide recommendations."""
        if not self.model:
            st.error("Gemini API not configured.")
            return None
        
        try:
            # Prepare the data for Gemini
            location_data = []
            for location in locations[:10]:  # Send top 10 candidates to Gemini
                location_info = location.to_dict()
                
                # Add nearby features if available
                if location.nearby_features:
                    location_info['nearby_features'] = location.nearby_features
                
                location_data.append(location_info)
            
            # Create an enhanced prompt for Gemini
            prompt = f'''
            I've conducted a site suitability analysis with the following criteria:
            
            Objective: {criteria.objective}
            
            Layers used:
            {json.dumps(criteria.layers, indent=2)}
            
            Distance requirements:
            {json.dumps(criteria.distance_requirements, indent=2)}
            
            Weights:
            {json.dumps(criteria.weights, indent=2)}
            
            Analysis summary:
            {analysis_summary}
            
            Here are the top candidate locations from my analysis, including real-world context and nearby features:
            {json.dumps(location_data, indent=2)}
            
            Based on this information, please:
            1. Select the top 3 most viable locations
            2. Provide a detailed explanation for each selection, referencing the specific neighborhood and nearby features
            3. Suggest potential development considerations for each location based on the surrounding context
            4. Compare the top 3 locations, highlighting the strengths and weaknesses of each
            5. Be specific about the neighborhoods or areas in Islamabad, Pakistan and use your knowledge of local context
            Format your response as a JSON object with these keys:
            - top_locations: Array of the 3 selected location IDs
            - explanations: Object with location IDs as keys and explanation text as values that references the real-world context
            - considerations: Object with location IDs as keys and development considerations as values
            - comparison: A comparative analysis of the three locations
            - overall_summary: Text summarizing your analysis that mentions the specific neighborhoods/areas
            
            The JSON should look like:
            {{
              "top_locations": [1, 5, 3],
              "explanations": {{
                "1": "Explanation for location 1 in [neighborhood/city]...",
                "5": "Explanation for location 5 in [neighborhood/city]...",
                "3": "Explanation for location 3 in [neighborhood/city]..."
              }},
              "considerations": {{
                "1": "Considerations for location 1...",
                "5": "Considerations for location 5...",
                "3": "Considerations for location 3..."
              }},
              "comparison": "Comparative analysis of the three locations...",
              "overall_summary": "Overall summary mentioning specific neighborhoods..."
            }}
            
            Return ONLY this JSON without additional text, markdown formatting, or explanations.
            '''
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.replace("```", "", 1)
                
            response_text = response_text.strip()
            
            # Parse the JSON response
            try:
                results = json.loads(response_text)
                return results
            except json.JSONDecodeError:
                st.error("Could not parse Gemini's response as JSON.")
                return {"error": "JSON parsing failed", "raw_response": response_text}
            
        except Exception as e:
            st.error(f"Error analyzing locations with Gemini: {e}")
            return {"error": str(e)}

def process_analysis_query(model, query, analysis_result):
    """Process a query about the analysis result with comprehensive context, 
    with Gemini acting as both a GIS expert and urban planner."""
    
    # Create a detailed context about the analysis
    context = {
        "objective": analysis_result.criteria.objective,
        "timestamp": analysis_result.timestamp.isoformat(),
        "total_locations": len(analysis_result.locations),
        "criteria_layers": analysis_result.criteria.layers,
        "distance_requirements": analysis_result.criteria.distance_requirements,
        "weights": analysis_result.criteria.weights,
        "avoid_layers": analysis_result.criteria.avoid
    }
    
    # Add top locations data
    top_locations = []
    for loc in analysis_result.get_top_locations(5):  # Get top 5 locations
        location_data = {
            "id": loc.location_id,
            "score": loc.suitability_score,
            "area_hectares": loc.area_hectares,
            "coordinates": [loc.latitude, loc.longitude],
            "address": loc.address if hasattr(loc, 'address') and loc.address else "Not available",
            "neighborhood": loc.neighborhood if hasattr(loc, 'neighborhood') and loc.neighborhood else "Not available",
            "city": loc.city if hasattr(loc, 'city') and loc.city else "Not available"
        }
        
        # Add nearby features if available
        if hasattr(loc, 'nearby_features') and loc.nearby_features:
            nearby = {}
            for layer, features in loc.nearby_features.items():
                feature_list = []
                for f in features[:3]:  # Take top 3 features per layer
                    feature_list.append({
                        "name": f.get('name', 'Unknown'),
                        "distance": f.get('distance', 0)
                    })
                nearby[layer] = feature_list
            location_data["nearby_features"] = nearby
            
        # Add explanations if available
        if hasattr(loc, 'explanation') and loc.explanation:
            location_data["explanation"] = loc.explanation
            
        if hasattr(loc, 'considerations') and loc.considerations:
            location_data["considerations"] = loc.considerations
            
        top_locations.append(location_data)
    
    context["top_locations"] = top_locations
    
    # Add Gemini analysis if available
    if hasattr(analysis_result, 'gemini_analysis') and analysis_result.gemini_analysis:
        context["gemini_analysis"] = {
            "overall_summary": analysis_result.gemini_analysis.get('overall_summary', ''),
            "comparison": analysis_result.gemini_analysis.get('comparison', '')
        }
    
    # Create a comprehensive prompt
    prompt = f"""
    You are an expert GIS developer and urban planner with local knowledge of Islamabad, Pakistan and extensive experience in land suitability analysis, 
    site selection, and urban development planning. You have deep knowledge of:
    
    1. GIS methodologies and spatial analysis techniques
    2. Urban planning principles and best practices
    3. Land use regulations and zoning considerations
    4. Infrastructure development requirements
    5. Environmental impact assessments
    6. Sustainable development practices
    7. Community engagement in urban planning
    8. Real estate development feasibility
    9. Transportation planning and accessibility
    10. Public utilities and services planning
    
    You're helping analyze a land suitability study. Here's the key information about the analysis:
    
    OBJECTIVE: {context['objective']}
    
    ANALYSIS DETAILS:
    - Analysis date: {context['timestamp']}
    - Found {context['total_locations']} suitable locations
    - Criteria layers used: {', '.join(context['criteria_layers'])}
    
    CRITERIA DETAILS:
    - Distance requirements: {context['distance_requirements']}
    - Weights used: {context['weights']}
    - Layers to avoid: {context['avoid_layers']}
    
    TOP LOCATIONS:
    """
    
    # Add top locations to the prompt
    for loc in context['top_locations']:
        prompt += f"""
        LOCATION {loc['id']} (Score: {loc['score']:.2f}, Area: {loc['area_hectares']:.2f} ha)
        - Coordinates: {loc['coordinates'][0]:.6f}, {loc['coordinates'][1]:.6f}
        - Address: {loc['address']}
        - Neighborhood: {loc['neighborhood']}
        - City: {loc['city']}
        """
        
        # Add nearby features if available
        if 'nearby_features' in loc:
            prompt += "- Nearby features:\n"
            for layer, features in loc['nearby_features'].items():
                feature_text = f"  * {layer.title()}: "
                feature_details = []
                for f in features:
                    feature_details.append(f"{f['name']} ({f['distance']:.0f}m)")
                prompt += feature_text + ", ".join(feature_details) + "\n"
        
        # Add explanations and considerations
        if 'explanation' in loc:
            prompt += f"- Analysis: {loc['explanation']}\n"
            
        if 'considerations' in loc:
            prompt += f"- Considerations: {loc['considerations']}\n"
    
    # Add Gemini analysis if available
    if 'gemini_analysis' in context:
        prompt += f"""
        OVERALL ANALYSIS:
        {context['gemini_analysis']['overall_summary']}
        
        COMPARATIVE ANALYSIS:
        {context['gemini_analysis']['comparison']}
        """
    
    # Expert knowledge context
    prompt += """
    As a GIS expert and urban planner for Islamabad, Pakistan, you should:
    
    1. Apply professional urban planning principles when answering questions
    2. Reference standard GIS methodologies and best practices
    3. Consider environmental, social, and economic sustainability factors
    4. Make connections to real-world development considerations beyond what's in the data
    5. Suggest additional analyses or data layers that could improve the study when relevant
    6. Provide technical explanations about GIS methods when appropriate
    7. Consider local context of Islamabad, Pakistan when making recommendations
    8. Relate site characteristics to potential development constraints or opportunities
    9. Comment on accessibility, transportation, and infrastructure implications
    10. Consider local regulations and planning frameworks in your responses
    11. Don't undermine the analysis by the app.
    You should respond not just as a data analyst, but as a seasoned professional who understands 
    the practical challenges and opportunities in urban development and site selection.
    """
    
    # Add user query
    prompt += f"""
    
    Based on the land suitability analysis and your expertise, please answer the following question:
    {query}
    
    Frame your answer in terms of professional GIS analysis and urban planning expertise. Don't talk about limitations too much.
    """
    
    # Generate response
    response = model.generate_content(prompt)
    return response.text

class GeoCodingService:
    """Service for geocoding operations."""
    
    def __init__(self, user_agent: str = "land_suitability_analysis_app"):
        """Initialize with user agent."""
        self.geolocator = Nominatim(user_agent=user_agent)
    
    def reverse_geocode(self, location: SuitableLocation) -> Dict:
        """Reverse geocode a location to get address information."""
        try:
            result = self.geolocator.reverse(f"{location.latitude}, {location.longitude}", exactly_one=True)
            
            if not result:
                return {
                    "address": "Address not found",
                    "neighborhood": "",
                    "suburb": "",
                    "city": "",
                    "state": ""
                }
            
            address = result.raw['address']
            
            # Extract useful information from address
            neighborhood = address.get('neighbourhood', '')
            suburb = address.get('suburb', '')
            city_district = address.get('city_district', '')
            city = address.get('city', address.get('town', ''))
            state = address.get('state', '')
            
            # Format a readable location description
            location_description = ", ".join(filter(None, [neighborhood, suburb, city_district, city, state]))
            
            return {
                "address": result.address,
                "location_description": location_description if location_description else "Location details not available",
                "neighborhood": neighborhood,
                "suburb": suburb,
                "city": city,
                "state": state
            }
        except Exception as e:
            st.warning(f"Error in reverse geocoding: {e}")
            return {
                "address": "Could not retrieve address",
                "neighborhood": "",
                "suburb": "",
                "city": "",
                "state": ""
            }
    
    def reverse_geocode_locations(self, locations: List[SuitableLocation]) -> None:
        """Reverse geocode a list of locations and update their attributes."""
        for location in locations:
            geocode_result = self.reverse_geocode(location)
            location.address = geocode_result["address"]
            location.neighborhood = geocode_result["neighborhood"]
            location.city = geocode_result["city"]

#############################################
# Spatial Analysis Services
#############################################

class SpatialAnalysisService:
    """Service for spatial analysis operations."""
    
    @staticmethod
    def create_analysis_grid(layers: Dict[str, gpd.GeoDataFrame], cell_size: float = 30) -> Optional[AnalysisGrid]:
        """Create a consistent analysis grid for all layers."""
        # First, ensure all layers are in WGS84 for consistent processing
        for name, gdf in layers.items():
            if gdf.crs is None:
                st.warning(f"Layer {name} has no CRS. Assuming WGS84.")
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
                layers[name] = gdf
        
        # Get study area bounds in WGS84
        all_bounds = []
        for gdf in layers.values():
            if not gdf.empty:
                bounds = gdf.total_bounds
                all_bounds.append(bounds)
        
        if not all_bounds:
            st.error("No valid bounds found in any layer")
            return None
        
        # Calculate overall bounds in WGS84
        bounds_array = np.array(all_bounds)
        xmin = bounds_array[:, 0].min()
        ymin = bounds_array[:, 1].min()
        xmax = bounds_array[:, 2].max()
        ymax = bounds_array[:, 3].max()
        
        # Calculate centroid for UTM zone selection
        center_lon = (xmin + xmax) / 2
        center_lat = (ymin + ymax) / 2
        
        # Determine UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone}" if center_lat >= 0 else f"EPSG:327{utm_zone}"
        
        # Create a temporary GeoDataFrame with the extent polygon in WGS84
        extent_poly = box(xmin, ymin, xmax, ymax)
        extent_gdf = gpd.GeoDataFrame(geometry=[extent_poly], crs="EPSG:4326")
        
        # Project the extent to UTM
        extent_gdf_utm = extent_gdf.to_crs(utm_crs)
        bounds_utm = extent_gdf_utm.total_bounds
        
        # Add buffer to ensure coverage
        buffer_size = cell_size * 2
        xmin_utm = bounds_utm[0] - buffer_size
        ymin_utm = bounds_utm[1] - buffer_size
        xmax_utm = bounds_utm[2] + buffer_size
        ymax_utm = bounds_utm[3] + buffer_size
        
        # Calculate grid dimensions
        width = int(np.ceil((xmax_utm - xmin_utm) / cell_size))
        height = int(np.ceil((ymax_utm - ymin_utm) / cell_size))
        
        # Create transform (from upper left corner)
        transform = rasterio.transform.from_origin(xmin_utm, ymax_utm, cell_size, cell_size)
        
        # Print debug information
        st.write("Analysis Grid Information:")
        st.write(f"WGS84 Bounds: {xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}")
        st.write(f"UTM Zone: {utm_zone}")
        st.write(f"UTM Bounds: {xmin_utm:.1f}, {ymin_utm:.1f}, {xmax_utm:.1f}, {ymax_utm:.1f}")
        st.write(f"Grid Dimensions: {width} x {height} cells")
        st.write(f"Cell Size: {cell_size} meters")
        
        return AnalysisGrid(transform, width, height, utm_crs, (xmin_utm, ymin_utm, xmax_utm, ymax_utm))
    
    @staticmethod
    def calculate_distance_raster(gdf: gpd.GeoDataFrame, grid: AnalysisGrid, cell_size: float = 30) -> Optional[np.ndarray]:
        """Calculate distance raster from features in the GeoDataFrame."""
        try:
            # Get geometry types present in the layer
            geom_types = set(gdf.geometry.type)
            
            # Create grid points for the raster
            xs = np.linspace(grid.transform.c, grid.transform.c + (grid.width - 1) * grid.transform.a, grid.width)
            ys = np.linspace(grid.transform.f, grid.transform.f + (grid.height - 1) * grid.transform.e, grid.height)
            grid_x, grid_y = np.meshgrid(xs, ys)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            
            # Create GeoDataFrame with grid points
            grid_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), 
                crs=gdf.crs
            )

            # For point geometries
            if geom_types.issubset({'Point', 'MultiPoint'}):
                # Handle MultiPoint geometries by converting to Point if needed
                if 'MultiPoint' in geom_types:
                    gdf = gdf.copy()
                    mask = gdf.geometry.type == 'MultiPoint'
                    gdf.loc[mask, 'geometry'] = gdf.loc[mask, 'geometry'].apply(lambda g: g.centroid)
                
                try:
                    # Use PyLUSAT's to_point function
                    distances = distance.to_point(
                        input_gdf=grid_gdf,
                        point_gdf=gdf,
                        method='euclidean'
                    )
                except Exception as e:
                    st.warning(f"PyLUSAT to_point failed: {e}. Using direct calculation.")
                    
                    # Fallback to direct calculation
                    distances_array = np.full(len(grid_gdf), np.inf)
                    
                    # Extract coordinates from grid points and target points
                    grid_coords = np.array([(p.x, p.y) for p in grid_gdf.geometry])
                    target_coords = np.array([(p.x, p.y) for p in gdf.geometry])
                    
                    # Use vectorized operations for better performance
                    for i in range(len(grid_coords)):
                        # Calculate squared distances to all points at once
                        squared_dists = (target_coords[:, 0] - grid_coords[i, 0])**2 + (target_coords[:, 1] - grid_coords[i, 1])**2
                        # Get minimum distance
                        distances_array[i] = np.sqrt(np.min(squared_dists)) if len(squared_dists) > 0 else np.inf
                    
                    # Convert to pandas Series to match PyLUSAT's output format
                    distances = pd.Series(distances_array)
            
            # For line geometries
            elif geom_types.issubset({'LineString', 'MultiLineString'}):
                # Use pylusat's to_line to calculate distances
                try:
                    distances = distance.to_line(grid_gdf, gdf, cellsize=cell_size, method='euclidean')
                except Exception as e:
                    st.warning(f"PyLUSAT to_line failed: {e}. Using alternative method.")
                    
                    # Rasterize the line geometries
                    rasterized = np.zeros((grid.height, grid.width), dtype=np.uint8)
                    shapes = [(geom, 1) for geom in gdf.geometry]
                    rasterized = rasterize(
                        shapes=shapes,
                        out=rasterized,
                        transform=grid.transform,
                        fill=0,
                        all_touched=True
                    )
                    
                    # Calculate distance transform
                    binary_image = (rasterized == 0).astype(np.uint8)
                    dist_transform = distance_transform_edt(binary_image) * cell_size
                    
                    # Flatten the distances to match the grid points
                    distances = pd.Series(dist_transform.flatten())
            
            # For polygon geometries or mixed types, rasterize and use distance transform
            else:
                # Rasterize the geometries
                rasterized = np.zeros((grid.height, grid.width), dtype=np.uint8)
                shapes = [(geom, 1) for geom in gdf.geometry]
                rasterized = rasterize(
                    shapes=shapes,
                    out=rasterized,
                    transform=grid.transform,
                    fill=0,
                    all_touched=True
                )
                
                # Calculate distance transform (distance to nearest non-zero cell)
                binary_image = (rasterized == 0).astype(np.uint8)
                dist_transform = distance_transform_edt(binary_image) * cell_size
                
                # Flatten the distances to match the grid points
                distances = pd.Series(dist_transform.flatten())
            
            # Reshape the distances to a 2D array (raster)
            distance_raster = distances.values.reshape((grid.height, grid.width))
            
            # Verify there are no NaN values in the raster
            if np.isnan(distance_raster).any():
                st.warning("NaN values detected in the distance raster, replacing with max distance")
                max_dist = np.nanmax(distance_raster)
                distance_raster = np.nan_to_num(distance_raster, nan=max_dist)

            return distance_raster
        
        except Exception as e:
            st.error(f"Error in distance calculation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    @staticmethod
    def reclassify_raster(raster_array: np.ndarray, reclassify_dict: Dict, nodata_value: float = -9999) -> np.ndarray:
        """Reclassify raster based on value ranges with continuous gradient."""
        output = np.full_like(raster_array, nodata_value, dtype=float)
        
        # Create a mask for valid data
        valid_mask = (raster_array != nodata_value)
        
        # For continuous gradient approach, we'll use this flag to determine if we're using the old method
        use_continuous_scale = True
        for (min_val, max_val) in reclassify_dict.keys():
            if isinstance(min_val, str) or isinstance(max_val, str):
                use_continuous_scale = False  # Fall back to old method if string values
                break
        
        if use_continuous_scale:
            # Extract distance requirement from reclassify_dict for continuous scaling
            try:
                # We'll assume there are typically two ranges in the dict: (0, dist_req) and (dist_req, inf)
                items = list(reclassify_dict.items())
                
                # Determine if this is an "avoid" layer (where higher distance is better)
                if len(items) >= 2:
                    is_avoid = items[0][1] < items[1][1]  # Check if first item value < second item value
                    distance_req = items[0][0][1]  # Get the distance requirement
                    
                    # Create continuous gradient based on distance
                    if is_avoid:  # For layers to avoid (higher distance is better)
                        # Start at 0 for d=0, increase to 100 as d approaches 2*distance_req
                        max_distance = 2 * distance_req
                        # Apply sigmoid-like scaling: slow start, rapid middle, slow end
                        output[valid_mask] = np.clip(100 * (raster_array[valid_mask] / max_distance)**1.5, 0, 100)
                    else:  # For layers to be near (lower distance is better)
                        # Start at 100 for d=0, decrease to 0 as d approaches distance_req
                        # Apply exponential decay formula: 100 * e^(-3d/distance_req)
                        output[valid_mask] = 100 * np.exp(-3 * raster_array[valid_mask] / distance_req)
                        
                    return output
                    
            except (IndexError, TypeError) as e:
                # Fall back to old method if any issue occurs
                st.warning(f"Using standard reclassification due to: {e}")
                use_continuous_scale = False
        
        # Fall back to original discrete classification approach
        if not use_continuous_scale:
            # Apply reclassification only to valid data
            for (min_val, max_val), new_val in reclassify_dict.items():
                try:
                    # Convert min_val to float, handling string inputs
                    if isinstance(min_val, str):
                        # Extract numerical value using regex
                        numbers = re.findall(r'\d+\.?\d*', min_val)
                        if numbers:
                            min_val_float = float(numbers[0])
                        else:
                            st.warning(f"Could not extract numerical value from '{min_val}', using 0")
                            min_val_float = 0.0
                    
                    else:
                        min_val_float = float(min_val)
                    
                    # Handle infinity specially
                    if max_val == float('inf') or max_val == 'inf':
                        mask = (raster_array >= min_val_float) & valid_mask
                    else:
                        # Convert max_val to float, handling string inputs
                        if isinstance(max_val, str):
                            if max_val.lower() == 'inf' or max_val.lower() == 'infinity':
                                max_val_float = float('inf')
                            else:
                                # Extract numerical value using regex
                                numbers = re.findall(r'\d+\.?\d*', max_val)
                                if numbers:
                                    max_val_float = float(numbers[0])
                                else:
                                    st.warning(f"Could not extract numerical value from '{max_val}', using infinity")
                                    max_val_float = float('inf')
                        else:
                            max_val_float = float(max_val)
                        
                        mask = (raster_array >= min_val_float) & (raster_array <= max_val_float) & valid_mask
                        
                    output[mask] = new_val
                    
                except Exception as e:
                    st.error(f"Error in reclassification: {e} for values min={min_val}, max={max_val}")
                    # Continue with next reclassification rule
                    continue
        
        return output
    
    @staticmethod
    def weighted_overlay(rasters: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
        """Perform weighted overlay of multiple rasters."""
        # Validate inputs
        if not rasters or not weights:
            st.error("No rasters or weights provided for overlay")
            return None
        
        # Check if we have the same number of rasters and weights
        if len(rasters) != len(weights):
            st.error(f"Number of rasters ({len(rasters)}) doesn't match number of weights ({len(weights)})")
            return None
        
        # Check if all rasters have the same shape
        shapes = [r.shape for r in rasters]
        unique_shapes = set(shapes)
        
        if len(unique_shapes) > 1:
            st.warning(f"Rasters have different dimensions: {unique_shapes}")
            
            # Find the smallest dimensions that will fit all rasters
            min_height = min(shape[0] for shape in shapes)
            min_width = min(shape[1] for shape in shapes)
            
            st.info(f"Resizing all rasters to {min_height} x {min_width}")
            
            # Resize all rasters to the smallest dimension
            resized_rasters = []
            for i, raster in enumerate(rasters):
                if raster.shape != (min_height, min_width):
                    # Simple resize by slicing (faster than resampling for this purpose)
                    resized = raster[:min_height, :min_width]
                    resized_rasters.append(resized)
                else:
                    resized_rasters.append(raster)
            
            rasters = resized_rasters
        
        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=float)
        weight_sum = weights.sum()
        
        if weight_sum == 0:
            st.error("Sum of weights is zero, cannot perform weighted overlay")
            return None
            
        weights = weights / weight_sum
        
        # Create output raster with the same shape as the input rasters
        output = np.zeros_like(rasters[0], dtype=float)
        
        # Create a mask to track valid data cells (cells where at least one raster has data)
        valid_mask = np.zeros_like(output, dtype=bool)
        
        # Apply weights
        for raster, weight in zip(rasters, weights):
            # Skip if weight is zero
            if weight == 0:
                continue
                
            # Create mask for nodata values
            raster_valid = (raster != -9999)
            
            # Update the overall valid mask
            valid_mask = valid_mask | raster_valid
            
            # Apply weight only to valid data
            output[raster_valid] += raster[raster_valid] * weight
        
        # Set nodata values in the output
        output[~valid_mask] = -9999
        
        return output
    
    @staticmethod
    def extract_suitable_locations(suitability_raster: np.ndarray, 
                                  grid: AnalysisGrid, 
                                  threshold_percentile: float = 95) -> List[SuitableLocation]:
        """Extract top suitable locations from the suitability raster."""
        try:
            # Create a mask for the highest suitability areas
            masked_raster = np.ma.masked_equal(suitability_raster, -9999)
            threshold = np.percentile(masked_raster.compressed(), threshold_percentile)
            high_suitability_mask = (masked_raster >= threshold) & (~np.ma.getmaskarray(masked_raster))
            
            # Find clusters of high suitability cells
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(high_suitability_mask)
            
            # Calculate size and average suitability of each cluster
            locations = []
            for i in range(1, num_features + 1):
                cluster_mask = labeled_array == i
                cluster_size = np.sum(cluster_mask)
                avg_suitability = np.mean(masked_raster[cluster_mask])
                
                # Find centroid of cluster
                rows, cols = np.where(cluster_mask)
                centroid_row = np.mean(rows)
                centroid_col = np.mean(cols)
                
                # Convert to UTM coordinates
                x = grid.transform.c + centroid_col * grid.transform.a
                y = grid.transform.f + centroid_row * grid.transform.e
                
                # Calculate area in hectares
                area_hectares = cluster_size * (30 ** 2) / 10000  # Assuming 30m cell size, convert to hectares
                
                # Create Point geometry
                point = Point(x, y)
                
                # Create a GeoDataFrame with the point in UTM coords
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs=grid.utm_crs)
                
                # Convert to WGS84 for lat/lon
                point_wgs84 = point_gdf.to_crs("EPSG:4326")
                lon = point_wgs84.geometry.x[0]
                lat = point_wgs84.geometry.y[0]
                
                # Create a SuitableLocation object
                location = SuitableLocation(
                    location_id=i,
                    geometry=point,
                    area_hectares=area_hectares,
                    suitability_score=float(avg_suitability),
                    latitude=lat,
                    longitude=lon
                )
                
                locations.append(location)
            
            # Sort locations by suitability score
            locations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            return locations
            
        except Exception as e:
            st.error(f"Error extracting locations: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return []


class VisualizationService:
    """Service for data visualization."""
    
    @staticmethod
    def visualize_raster(raster_array: np.ndarray, 
                        transform, 
                        title: str, 
                        cmap: str = 'viridis', 
                        nodata: float = -9999, 
                        boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> plt.Figure:
        """Create a visualization of a raster."""
        # Create a masked array to handle nodata values
        masked_array = np.ma.masked_where(raster_array == nodata, raster_array)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Creating a custom colormap that goes from red to green
        colors = [(0.8, 0, 0), (1, 1, 0), (0, 0.8, 0)]  # Red to Yellow to Green
        custom_cmap = LinearSegmentedColormap.from_list('custom_suitability', colors)
        
        # Plot the raster
        img = ax.imshow(masked_array, cmap=custom_cmap)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Suitability Score')
        
        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Remove axes
        ax.set_axis_off()
        
        # Overlay boundary if provided
        if boundary_gdf is not None:
            for geom in boundary_gdf.geometry:
                if geom.type == 'Polygon':
                    mpl_poly = MplPolygon(np.array(geom.exterior.coords), fill=False, edgecolor='blue', linewidth=2)
                    ax.add_patch(mpl_poly)
                elif geom.type == 'MultiPolygon':
                    for part in geom.geoms:
                        mpl_poly = MplPolygon(np.array(part.exterior.coords), fill=False, edgecolor='blue', linewidth=2)
                        ax.add_patch(mpl_poly)
        
        return fig
    
    @staticmethod
    def create_locations_map(locations: List[SuitableLocation], 
                            top_location_ids: List[int] = None, 
                            boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> folium.Map:
        """Create an interactive folium map with the suitable locations."""
        # Filter for top locations if specified
        if top_location_ids:
            top_locations = [loc for loc in locations if loc.location_id in top_location_ids]
        else:
            top_locations = locations[:3]  # Default to top 3
        
        if not top_locations:
            return None
        
        # Calculate map center
        center_lat = sum(loc.latitude for loc in top_locations) / len(top_locations)
        center_lon = sum(loc.longitude for loc in top_locations) / len(top_locations)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                       tiles='CartoDB positron', control_scale=True)
        
        # Add boundary if provided
        if boundary_gdf is not None:
            # Convert any date fields to strings to avoid JSON serialization issues
            boundary_gdf_wgs84 = boundary_gdf.to_crs(epsg=4326).copy()
            
            # Convert date columns to strings
            for col in boundary_gdf_wgs84.columns:
                if col != boundary_gdf_wgs84._geometry_column_name:
                    # Check if column contains dates
                    if boundary_gdf_wgs84[col].dtype.kind == 'M' or \
                       (len(boundary_gdf_wgs84) > 0 and any(isinstance(x, (date, datetime)) 
                                                         for x in boundary_gdf_wgs84[col].dropna().head(1))):
                        boundary_gdf_wgs84[col] = boundary_gdf_wgs84[col].apply(
                            lambda x: str(x) if x is not None else None
                        )
            
            folium.GeoJson(
                data=boundary_gdf_wgs84.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'blue',
                    'weight': 2,
                    'dashArray': '5, 5'
                },
                name='Boundary'
            ).add_to(m)
        
        # Add top locations with enhanced information
        for location in top_locations:
            # Create simplified HTML for popup (avoiding complex data structures)
            html = f'''
            <div style="font-family: Arial; max-width: 350px;">
                <h3 style="color: #4A90E2;">Location {location.location_id}</h3>
                <div style="margin-bottom: 10px;">
                    <strong>Area:</strong> {location.area_hectares:.2f} hectares<br>
                    <strong>Suitability Score:</strong> {location.suitability_score:.2f}<br>
                    <strong>Coordinates:</strong> {location.latitude:.6f}, {location.longitude:.6f}
                </div>
            '''
            
            # Add address information if available - ensure strings are used
            if location.address:
                html += f'''
                <div style="margin-bottom: 10px;">
                    <strong>Address:</strong> {str(location.address)}<br>
                    <strong>Neighborhood:</strong> {str(location.neighborhood or "Not available")}<br>
                    <strong>City:</strong> {str(location.city or "Not available")}<br>
                </div>
                '''
            
            # Add nearby features if available - ensure all values are properly stringified
            if location.nearby_features:
                html += "<div style='margin-bottom: 10px;'><h4 style='margin-bottom: 5px;'>Nearby Features:</h4><ul style='padding-left: 20px; margin-top: 5px;'>"
                for layer, features in location.nearby_features.items():
                    if features:
                        html += f"<li><strong>{str(layer.title())}</strong>: "
                        feature_list = []
                        for f in features[:3]:
                            # Ensure we're using primitive types that are JSON serializable
                            name = str(f.get('name', ''))
                            distance = float(f.get('distance', 0))
                            feature_list.append(f"{name} ({distance:.0f}m)")
                        html += ", ".join(feature_list)
                        html += "</li>"
                html += "</ul></div>"
            
            # Add explanation and considerations if available - ensure they're strings
            if hasattr(location, 'explanation') and location.explanation:
                html += f'''
                <div style="margin-bottom: 10px;">
                    <h4 style="margin-bottom: 5px;">Analysis:</h4>
                    <p style="margin-top: 5px;">{str(location.explanation)}</p>
                </div>
                '''
            
            if hasattr(location, 'considerations') and location.considerations:
                html += f'''
                <div>
                    <h4 style="margin-bottom: 5px;">Development Considerations:</h4>
                    <p style="margin-top: 5px;">{str(location.considerations)}</p>
                </div>
                '''
            
            html += "</div>"
            
            iframe = folium.IFrame(html=html, width=370, height=400)
            popup = folium.Popup(iframe)
            
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=popup,
                icon=folium.Icon(color='red', icon='star'),
                tooltip=f"Location {location.location_id} - Score: {location.suitability_score:.2f}"
            ).add_to(m)
        
        # Add all other locations with simpler markers
        other_locations = [loc for loc in locations if loc.location_id not in [l.location_id for l in top_locations]]
        for location in other_locations[:20]:  # Limit to 20 to avoid clutter
            folium.CircleMarker(
                location=[location.latitude, location.longitude],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                tooltip=f"Location {location.location_id} - Score: {location.suitability_score:.2f}"
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

#############################################
# PDF Report Generation Service
#############################################

class PDFReportService:
    """Service for generating PDF reports of analysis results."""
    
    def __init__(self, result: AnalysisResult, boundary_gdf: Optional[gpd.GeoDataFrame] = None):
        """Initialize with analysis results."""
        self.result = result
        self.boundary_gdf = boundary_gdf
        self.styles = getSampleStyleSheet()
        self.initialize_styles()
    
    def initialize_styles(self):
        """Initialize custom styles for the PDF report."""
        # Add custom styles - with checks to avoid duplicates
        style_definitions = [
            ('ReportTitle', {
                'parent': self.styles['Heading1'],
                'fontSize': 20,
                'alignment': TA_CENTER,
                'spaceAfter': 12
            }),
            ('Subtitle', {
                'parent': self.styles['Heading2'],
                'fontSize': 16,
                'spaceBefore': 12,
                'spaceAfter': 6
            }),
            ('Body', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'spaceBefore': 6,
                'spaceAfter': 6
            }),
            ('Location', {
                'parent': self.styles['Normal'],
                'fontSize': 12,
                'textColor': colors.navy,
                'spaceBefore': 6,
                'spaceAfter': 6
            })
        ]
        
        # Add styles if they don't already exist
        for style_name, style_props in style_definitions:
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(name=style_name, **style_props))
    
    def generate_report(self, output_path: str) -> bool:
        """Generate a PDF report of the analysis results."""
        # List to keep track of temporary files
        temp_files = []
        
        try:
            # Create a PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=1*cm,
                leftMargin=1*cm,
                topMargin=1.5*cm,
                bottomMargin=1.5*cm
            )
            
            # Build the content
            content = []
            
            # Add title
            title = Paragraph("Land Suitability Analysis Report", self.styles['ReportTitle'])
            content.append(title)
            
            # Add date - using string format to avoid JSON serialization issues
            date_text = Paragraph(f"Generated on: {self.result.timestamp.strftime('%Y-%m-%d %H:%M')}", self.styles['Body'])
            content.append(date_text)
            content.append(Spacer(1, 0.5*cm))
            
            # Add objective
            objective = Paragraph("Analysis Objective", self.styles['Subtitle'])
            content.append(objective)
            objective_text = Paragraph(self.result.criteria.objective, self.styles['Body'])
            content.append(objective_text)
            content.append(Spacer(1, 0.5*cm))
            
            # Add criteria summary
            criteria = Paragraph("Analysis Criteria", self.styles['Subtitle'])
            content.append(criteria)
            
            # Layers and weights table
            layers_data = [['Layer', 'Distance Requirement', 'Weight', 'Avoid?']]
            for layer in self.result.criteria.layers:
                distance = self.result.criteria.distance_requirements.get(layer, '')
                weight = self.result.criteria.weights.get(layer, '')
                avoid = "Yes" if layer in self.result.criteria.avoid else "No"
                layers_data.append([layer, f"{distance} meters", weight, avoid])
            
            layers_table = Table(layers_data, colWidths=[4*cm, 4*cm, 2*cm, 2*cm])
            layers_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            content.append(layers_table)
            content.append(Spacer(1, 0.5*cm))
            
            # Add individual layer rasters (NEW SECTION)
            if hasattr(self.result, 'named_rasters') and self.result.named_rasters:
                layer_rasters_title = Paragraph("Individual Layer Analysis", self.styles['Subtitle'])
                content.append(layer_rasters_title)
                content.append(Paragraph("Each layer was analyzed individually to create suitability surfaces before combining them.", self.styles['Body']))
                content.append(Spacer(1, 0.3*cm))
                
                # Create a plot for each raster
                for layer_name, raster_info in self.result.named_rasters.items():
                    # Create a subtitle for the layer
                    avoid_text = "avoid" if raster_info['avoid'] else "proximity to"
                    layer_title = Paragraph(f"{layer_name.title()} ({avoid_text})", self.styles['Location'])
                    content.append(layer_title)
                    
                    # Add layer details
                    distance_req = raster_info['distance_req']
                    weight = raster_info['weight']
                    layer_details = Paragraph(
                        f"Distance requirement: {distance_req}m | Weight: {weight}/10", 
                        self.styles['Body']
                    )
                    content.append(layer_details)
                    
                    # Generate visualization for this layer's raster
                    layer_map_title = f"{layer_name.title()} Suitability"
                    if raster_info['avoid']:
                        layer_map_title += " (higher values = further away)"
                    else:
                        layer_map_title += " (higher values = closer)"
                    
                    fig = VisualizationService.visualize_raster(
                        raster_info['raster'],
                        self.result.grid.transform,
                        layer_map_title,
                        boundary_gdf=self.boundary_gdf
                    )
                    
                    # Save figure to a temporary file
                    fd, temp_map_path = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                    temp_files.append(temp_map_path)
                    
                    try:
                        fig.savefig(temp_map_path, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        
                        if os.path.exists(temp_map_path) and os.path.getsize(temp_map_path) > 0:
                            img = Image(temp_map_path, width=16*cm, height=10*cm)
                            content.append(img)
                        else:
                            content.append(Paragraph(f"Could not generate map for {layer_name}.", self.styles['Body']))
                    except Exception as e:
                        content.append(Paragraph(f"Error generating map for {layer_name}: {str(e)}", self.styles['Body']))
                    
                    content.append(Spacer(1, 0.5*cm))
            
            # Add suitability map
            map_title = Paragraph("Final Suitability Map", self.styles['Subtitle'])
            content.append(map_title)
            
            # Generate map image
            fig = VisualizationService.visualize_raster(
                self.result.suitability_raster,
                self.result.grid.transform,
                "Land Suitability Analysis",
                boundary_gdf=self.boundary_gdf
            )
            
            # Save figure to a temporary file (keep it open until the end)
            fd, temp_map_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)  # Close the file descriptor but keep the file
            temp_files.append(temp_map_path)  # Track the temp file
            
            try:
                fig.savefig(temp_map_path, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Check if file exists and has content
                if os.path.exists(temp_map_path) and os.path.getsize(temp_map_path) > 0:
                    img = Image(temp_map_path, width=16*cm, height=12*cm)
                    content.append(img)
                else:
                    st.warning(f"Map image file is empty or not accessible: {temp_map_path}")
                    map_note = Paragraph(
                        "Note: The suitability map could not be included in this PDF. Please refer to the web application.",
                        self.styles['Body']
                    )
                    content.append(map_note)
            except Exception as e:
                st.error(f"Error saving map image: {e}")
                map_note = Paragraph(
                    "Note: The suitability map could not be included in this PDF. Please refer to the web application.",
                    self.styles['Body']
                )
                content.append(map_note)
            
            content.append(Spacer(1, 0.5*cm))
            
            # Add top locations section
            locations_title = Paragraph("Top Suitable Locations", self.styles['Subtitle'])
            content.append(locations_title)
            
            # Get top 3 locations
            top_locations = self.result.get_top_locations(3)
            
            # If we have Gemini analysis, use its top locations instead
            if self.result.gemini_analysis and 'top_locations' in self.result.gemini_analysis:
                top_ids = self.result.gemini_analysis['top_locations']
                top_locations = [loc for loc in self.result.locations if loc.location_id in top_ids]
            
            # Add the interactive map image - skip this as it causes issues
            # Instead, just add a note about the interactive map
            map_note = Paragraph(
                "Note: An interactive map is available in the web application.",
                self.styles['Body']
            )
            content.append(map_note)
            content.append(Spacer(1, 0.5*cm))
            
            # Add details for each top location
            for location in top_locations:
                location_title = Paragraph(f"Location {location.location_id}", self.styles['Location'])
                content.append(location_title)
                
                details = [
                    f"Area: {location.area_hectares:.2f} hectares",
                    f"Suitability Score: {location.suitability_score:.2f}",
                    f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}"
                ]
                
                if location.address:
                    details.append(f"Address: {str(location.address)}")
                
                if location.neighborhood:
                    details.append(f"Neighborhood: {str(location.neighborhood)}")
                
                if location.city:
                    details.append(f"City: {str(location.city)}")
                
                # Convert details to paragraphs
                for detail in details:
                    p = Paragraph(detail, self.styles['Body'])
                    content.append(p)
                
                # Add explanation if available from Gemini
                if location.explanation:
                    explanation_title = Paragraph("Analysis:", self.styles['Body'])
                    content.append(explanation_title)
                    explanation = Paragraph(str(location.explanation), self.styles['Body'])
                    content.append(explanation)
                
                # Add considerations if available from Gemini
                if location.considerations:
                    consid_title = Paragraph("Development Considerations:", self.styles['Body'])
                    content.append(consid_title)
                    consid = Paragraph(str(location.considerations), self.styles['Body'])
                    content.append(consid)
                
                # Add nearby features if available
                if location.nearby_features:
                    nearby_title = Paragraph("Nearby Features:", self.styles['Body'])
                    content.append(nearby_title)
                    
                    for layer, features in location.nearby_features.items():
                        if features:
                            feature_text = f"<b>{str(layer).title()}:</b> "
                            feature_details = []
                            for f in features[:3]:
                                name = str(f.get('name', ''))
                                distance = float(f.get('distance', 0))
                                feature_details.append(f"{name} ({distance:.0f}m)")
                            feature_text += ", ".join(feature_details)
                            feature_para = Paragraph(feature_text, self.styles['Body'])
                            content.append(feature_para)
                
                content.append(Spacer(1, 0.5*cm))
            
            # Add comparison section if available from Gemini
            if self.result.gemini_analysis and 'comparison' in self.result.gemini_analysis:
                comparison_title = Paragraph("Comparative Analysis of Top Locations", self.styles['Subtitle'])
                content.append(comparison_title)
                
                comparison_text = Paragraph(str(self.result.gemini_analysis['comparison']), self.styles['Body'])
                content.append(comparison_text)
                content.append(Spacer(1, 0.5*cm))
            
            # Add overall summary if available from Gemini
            if self.result.gemini_analysis and 'overall_summary' in self.result.gemini_analysis:
                summary_title = Paragraph("Overall Summary", self.styles['Subtitle'])
                content.append(summary_title)
                
                summary_text = Paragraph(str(self.result.gemini_analysis['overall_summary']), self.styles['Body'])
                content.append(summary_text)
            
            # Build the document
            doc.build(content)
            
            # Success - return true
            return True
            
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as cleanup_error:
                    st.warning(f"Could not delete temporary file {temp_file}: {cleanup_error}")

class SuitabilityAnalysisService:
    """Main service for performing land suitability analysis."""
    
    def __init__(self, db_service: DatabaseService, gemini_service: Optional[GeminiService] = None):
        """Initialize with required services."""
        self.db_service = db_service
        self.gemini_service = gemini_service
        self.geocoding_service = GeoCodingService()
    
    def analyze(self, criteria: AnalysisCriteria, cell_size: float = 30) -> Optional[AnalysisResult]:
        """Perform complete suitability analysis based on criteria."""
        try:
            # Fetch layers from database
            layers = self.db_service.fetch_layers(criteria.layers)
            
            if not layers:
                st.error("No layers could be loaded from the database.")
                return None
            
            # Create analysis grid
            grid = SpatialAnalysisService.create_analysis_grid(layers, cell_size)
            
            if grid is None:
                st.error("Could not create analysis grid.")
                return None
            
            # Process each layer
            processed_rasters = []
            raster_weights = []
            # Store named reclassified rasters for reporting
            named_rasters = {}
            
            for layer_name in criteria.layers:
                if layer_name not in layers:
                    st.warning(f"Layer {layer_name} not available, skipping.")
                    continue
                    
                # Get distance requirement and weight
                distance_req = criteria.distance_requirements.get(layer_name, 1000)
                weight = criteria.weights.get(layer_name, 5)
                avoid = layer_name in criteria.avoid
                
                # Project layer to UTM
                layer_utm = layers[layer_name].to_crs(grid.utm_crs)
                
                # Calculate distance raster
                distance_raster = SpatialAnalysisService.calculate_distance_raster(
                    layer_utm,
                    grid,
                    cell_size
                )
                
                if distance_raster is None:
                    continue
                
                # Reclassify the distance raster based on criteria
                if avoid:
                    # For layers to avoid (higher distance is better)
                    reclass_dict = {
                        (0, distance_req): 0,         # Too close (unsuitable)
                        (distance_req, float('inf')): 100  # Far enough (suitable)
                    }
                else:
                    # For layers to be near (lower distance is better)
                    reclass_dict = {
                        (0, distance_req): 100,        # Within range (suitable)
                        (distance_req, float('inf')): 0   # Too far (unsuitable)
                    }
                    
                reclassified = SpatialAnalysisService.reclassify_raster(distance_raster, reclass_dict)
                
                # Store the reclassified raster with its name
                named_rasters[layer_name] = {
                    'raster': reclassified,
                    'weight': weight,
                    'avoid': avoid,
                    'distance_req': distance_req
                }
                
                # Add to collection for overlay
                processed_rasters.append(reclassified)
                raster_weights.append(weight)
            
            # Perform weighted overlay
            if not processed_rasters:
                st.error("No rasters could be processed for analysis.")
                return None
                
            final_suitability = SpatialAnalysisService.weighted_overlay(processed_rasters, raster_weights)
            
            # Rescale to 0-100
            valid_mask = (final_suitability != -9999)
            if np.any(valid_mask):
                min_val = np.min(final_suitability[valid_mask])
                max_val = np.max(final_suitability[valid_mask])
                
                if max_val > min_val:
                    final_suitability[valid_mask] = (
                        (final_suitability[valid_mask] - min_val) / (max_val - min_val) * 100
                    )
            
            # Extract suitable locations
            locations = SpatialAnalysisService.extract_suitable_locations(
                final_suitability, 
                grid, 
                threshold_percentile=95
            )
            
            if not locations:
                st.warning("No suitable locations found meeting the criteria.")
                return None
            
            # Add geocoding information
            self.geocoding_service.reverse_geocode_locations(locations[:10])  # Only geocode top 10
            
            # Add nearby features information
            nearby_features = self.db_service.get_nearby_features(locations[:10])
            for location in locations[:10]:
                location_id = str(location.location_id)
                if location_id in nearby_features:
                    location.nearby_features = nearby_features[location_id]
            
            # Run Gemini analysis if available
            gemini_analysis = None
            if self.gemini_service:
                analysis_summary = f"Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M')}. Found {len(locations)} potential sites using {len(processed_rasters)} criteria layers."
                gemini_analysis = self.gemini_service.analyze_locations(locations, criteria, analysis_summary)
                
                # Update location objects with Gemini analysis
                if gemini_analysis and 'explanations' in gemini_analysis and 'considerations' in gemini_analysis:
                    for location in locations:
                        location_id = str(location.location_id)
                        if location_id in gemini_analysis['explanations']:
                            location.explanation = gemini_analysis['explanations'][location_id]
                        if location_id in gemini_analysis['considerations']:
                            location.considerations = gemini_analysis['considerations'][location_id]
            
            # Update AnalysisResult to include named rasters
            result = AnalysisResult(
                suitability_raster=final_suitability,
                grid=grid,
                locations=locations,
                criteria=criteria,
                gemini_analysis=gemini_analysis
            )
            
            # Add named rasters to a custom attribute
            result.named_rasters = named_rasters
            
            return result
            
        except Exception as e:
            st.error(f"Error in suitability analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
        
#############################################
# Application UI Components
#############################################

class UIComponents:
    """UI components for the Streamlit application."""
    
    @staticmethod
    def render_sidebar(db_service: Optional[DatabaseService] = None):
        """Render the sidebar UI."""
        st.sidebar.header("Configuration")
        
        # Database connection section
        st.sidebar.subheader("Database Connection")
        
        # Use session state to persist form values
        if 'db_host' not in st.session_state:
            st.session_state.db_host = "localhost"
        if 'db_port' not in st.session_state:
            st.session_state.db_port = "5432"
        if 'db_name' not in st.session_state:
            st.session_state.db_name = "gis"
        if 'db_user' not in st.session_state:
            st.session_state.db_user = "postgres"
        if 'db_password' not in st.session_state:
            st.session_state.db_password = ""
        
        db_host = st.sidebar.text_input("Host", value=st.session_state.db_host, key="db_host_input")
        db_port = st.sidebar.text_input("Port", value=st.session_state.db_port, key="db_port_input")
        db_name = st.sidebar.text_input("Database", value=st.session_state.db_name, key="db_name_input")
        db_user = st.sidebar.text_input("Username", value=st.session_state.db_user, key="db_user_input")
        db_password = st.sidebar.text_input("Password", type="password", value=st.session_state.db_password, key="db_password_input")
        
        # Gemini API configuration
        st.sidebar.subheader("Gemini API Configuration")
        
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = ""
        
        gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, key="gemini_api_key_input")
        
        # Connect button
        connect_pressed = st.sidebar.button("Connect", type="primary")
        
        if connect_pressed:
            # Update session state
            st.session_state.db_host = db_host
            st.session_state.db_port = db_port
            st.session_state.db_name = db_name
            st.session_state.db_user = db_user
            st.session_state.db_password = db_password
            st.session_state.gemini_api_key = gemini_api_key
            
            # Create database config and connect
            db_config = DatabaseConfig(db_host, db_port, db_name, db_user, db_password)
            
            if 'db_service' not in st.session_state:
                st.session_state.db_service = DatabaseService(db_config)
            else:
                st.session_state.db_service.config = db_config
            
            if st.session_state.db_service.connect():
                st.sidebar.success("Connected to database!")
                
                # Setup Gemini if API key provided
                if gemini_api_key:
                    try:
                        if 'gemini_service' not in st.session_state:
                            st.session_state.gemini_service = GeminiService(gemini_api_key)
                        else:
                            st.session_state.gemini_service.api_key = gemini_api_key
                        
                        if st.session_state.gemini_service.connect():
                            st.sidebar.success("Gemini API configured successfully!")
                    except Exception as e:
                        st.sidebar.error(f"Error setting up Gemini API: {e}")
                else:
                    st.sidebar.warning("Gemini API key not provided. Some features will be disabled.")
        
        # Advanced options
        with st.sidebar.expander("Advanced Options", expanded=False):
            if 'cell_size' not in st.session_state:
                st.session_state.cell_size = 30
            
            st.session_state.cell_size = st.number_input(
                "Cell size (meters)",
                min_value=10,
                max_value=100,
                value=st.session_state.cell_size,
                step=5,
                help="Size of the grid cells in meters. Smaller values provide more detailed analysis but increase computation time."
            )
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "This Land Suitability Analysis Tool helps urban planners and developers "
            "identify optimal locations for development based on multiple spatial criteria. "
            "Created using PyLUSAT, Streamlit, and Google's Gemini AI."
        )
        
        # Return the services
        return {
            'db_service': st.session_state.get('db_service'),
            'gemini_service': st.session_state.get('gemini_service')
        }
    
    @staticmethod
    def render_analysis_tab(db_service: Optional[DatabaseService] = None, 
                           gemini_service: Optional[GeminiService] = None):
        """Render the analysis tab UI."""
        st.header("Site Suitability Analysis", divider="blue")
        
        if not db_service or not db_service.engine:
            st.warning("Please connect to the database first using the sidebar.")
            return
        
        st.subheader("Define Suitability Criteria")
        
        # Get available layers
        available_layers = db_service.get_available_layers()
        
        # Two methods for defining criteria: Gemini AI or manual
        criteria_tab1, criteria_tab2 = st.tabs(["Define with AI", "Define Manually"])
        
        with criteria_tab1:
            if not gemini_service or not gemini_service.model:
                st.warning("Gemini API not configured. Please add an API key in the sidebar.")
            else:
                st.markdown('''
                Describe what you're looking for in plain English, and the AI will convert your description into specific criteria.
                
                **Example:** "I need to find a suitable location for a new residential area that should be within 2km of schools, 5km of healthcare facilities, and 1km of roads. It should also be at least 500m away from railways for safety. Proximity to schools is most important, followed by healthcare."
                ''')
                
                criteria_text = st.text_area(
                    "Describe your site suitability criteria in natural language",
                    value="I need to find a suitable location for a new residential area that should be within 2km of schools, 5km of healthcare facilities, and 1km of roads. It should also be at least 500m away from railways for safety. Proximity to schools is most important, followed by healthcare.",
                    height=150
                )
                
                if st.button("Analyze Criteria with AI", type="primary", key="analyze_criteria_btn"):
                    with st.spinner("Analyzing criteria with AI..."):
                        criteria = gemini_service.analyze_criteria(criteria_text)
                        
                        if criteria:
                            st.session_state.criteria = criteria
                            st.success("Criteria analyzed successfully!")
                            
                            # Display the identified criteria
                            st.subheader("Identified Criteria")
                            st.markdown(f"**Objective:** {criteria.objective}")
                            
                            # Display layers, distance requirements, and weights
                            criteria_df = pd.DataFrame({
                                "Layer": criteria.layers,
                                "Distance Requirement (m)": [criteria.distance_requirements.get(layer, "-") for layer in criteria.layers],
                                "Weight (1-100)": [criteria.weights.get(layer, "-") for layer in criteria.layers],
                                "Avoid?": [layer in criteria.avoid for layer in criteria.layers]
                            })
                            
                            st.dataframe(criteria_df, use_container_width=True)
                        else:
                            st.error("Failed to analyze criteria. Please try again with a more specific description.")
        
        with criteria_tab2:
            st.markdown("Manually select layers and define criteria for each layer.")
            
            # Display available layers
            st.info(f"Available layers: {', '.join(available_layers)}")
            
            # Manual layer selection
            selected_layers = st.multiselect("Select layers to include", available_layers)
            
            if selected_layers:
                st.subheader("Distance Requirements and Weights")
                
                manual_distance_reqs = {}
                manual_weights = {}
                manual_avoid_layers = []
                
                # Create columns for each criterion
                for layer in selected_layers:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        distance = st.number_input(f"Distance (m) for {layer}", min_value=100, max_value=10000, value=1000, step=100)
                        manual_distance_reqs[layer] = distance
                    
                    with col2:
                        weight = st.slider(f"Weight for {layer}", min_value=1, max_value=10, value=5)
                        manual_weights[layer] = weight
                    
                    with col3:
                        avoid = st.checkbox(f"Avoid {layer} (distance is minimum)")
                        if avoid:
                            manual_avoid_layers.append(layer)
                
                # Objective field
                objective = st.text_area("Analysis Objective", value="Manual site suitability analysis")
                
                # Create criteria button
                if st.button("Set Manual Criteria", type="primary"):
                    st.session_state.criteria = AnalysisCriteria(
                        layers=selected_layers,
                        distance_requirements=manual_distance_reqs,
                        weights=manual_weights,
                        avoid=manual_avoid_layers,
                        objective=objective
                    )
                    
                    st.success("Manual criteria set!")
                    
                    # Show the criteria as a table
                    criteria_df = pd.DataFrame({
                        "Layer": selected_layers,
                        "Distance Requirement (m)": [manual_distance_reqs.get(layer, "-") for layer in selected_layers],
                        "Weight (1-100)": [manual_weights.get(layer, "-") for layer in selected_layers],
                        "Avoid?": [layer in manual_avoid_layers for layer in selected_layers]
                    })
                    
                    st.dataframe(criteria_df, use_container_width=True)
        
        # Run analysis button
        st.markdown("---")
        
        if 'criteria' in st.session_state:
            st.success("Criteria are defined and ready for analysis.")
            
            if st.button("Run Suitability Analysis", type="primary", key="run_analysis_btn"):
                with st.spinner("Running analysis..."):
                    # Create analysis service
                    analysis_service = SuitabilityAnalysisService(db_service, gemini_service)
                    
                    # Run analysis
                    result = analysis_service.analyze(st.session_state.criteria, st.session_state.cell_size)
                    
                    if result:
                        st.session_state.analysis_result = result
                        
                        # Try to load boundary
                        try:
                            boundary = db_service.load_boundary("islamabad", result.grid.utm_crs)
                            st.session_state.boundary = boundary
                        except Exception as e:
                            st.warning(f"Could not load boundary: {e}")
                            st.session_state.boundary = None
                        
                        st.success("Analysis completed successfully!")
                        st.balloons()
                    else:
                        st.error("Analysis failed. Please check the logs for details.")
        else:
            st.info("Please define criteria before running the analysis.")
    
    @staticmethod
    def render_results_tab():
        """Render the results tab UI."""
        st.header("Analysis Results", divider="blue")
        
        if 'analysis_result' not in st.session_state:
            st.info("No analysis results available. Please run an analysis first.")
            return
        
        result = st.session_state.analysis_result
        boundary = st.session_state.get('boundary')
        
        # Overview section
        st.subheader("Overview")
        
        # Display basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Objective:** {result.criteria.objective}")
            st.markdown(f"**Analysis Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Number of Suitable Locations:** {len(result.locations)}")
        
        with col2:
            st.markdown(f"**Number of Criteria Layers:** {len(result.criteria.layers)}")
            layers_text = ", ".join(result.criteria.layers)
            st.markdown(f"**Layers Used:** {layers_text}")
        
        # Map section
        st.subheader("Suitability Map")
        
        # Create and display the visualization
        fig = VisualizationService.visualize_raster(
            result.suitability_raster,
            result.grid.transform,
            "Land Suitability Analysis",
            boundary_gdf=boundary
        )
        
        st.pyplot(fig)
        
        # Add map download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Map Image",
            data=buf,
            file_name="suitability_map.png",
            mime="image/png"
        )
        
        # Locations section
        st.subheader("Suitable Locations")
        
        # Create a DataFrame for easier display
        locations_data = []
        for loc in result.locations:
            locations_data.append({
                "ID": loc.location_id,
                "Area (ha)": loc.area_hectares,
                "Suitability Score": loc.suitability_score,
                "Latitude": loc.latitude,
                "Longitude": loc.longitude,
                "Neighborhood": loc.neighborhood or "N/A",
                "City": loc.city or "N/A"
            })
        
        locations_df = pd.DataFrame(locations_data)
        
        # Display the locations table
        st.dataframe(locations_df, use_container_width=True)
        
        # Add download buttons for locations data
        csv = locations_df.to_csv(index=False)
        st.download_button(
            label="Download Locations CSV",
            data=csv,
            file_name="suitable_locations.csv",
            mime="text/csv"
        )
        
        # Interactive map of locations
        st.subheader("Interactive Map")
        
        # Get top location IDs from Gemini analysis if available
        top_location_ids = None
        if result.gemini_analysis and 'top_locations' in result.gemini_analysis:
            top_location_ids = result.gemini_analysis['top_locations']
        
        try:
            # Create the map
            m = VisualizationService.create_locations_map(
                result.locations,
                top_location_ids,
                boundary
            )
            
            # Save to a temporary file to avoid JSON serialization issues
            if m:
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp:
                        m.save(temp.name)
                        temp_path = temp.name
                    
                    # Read the HTML file directly instead of using _repr_html_()
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    
                    # Display the map
                    st.components.v1.html(map_html, height=500)
                    
                    # Create a download button
                    st.download_button(
                        label="Download Interactive Map",
                        data=map_html,
                        file_name="suitability_map.html",
                        mime="text/html"
                    )
                finally:
                    # Clean up the temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                st.warning("Could not create interactive map.")
        except Exception as e:
            st.error(f"Error displaying interactive map: {e}")
            st.warning("Try running the analysis again with updated data.")
        
        # Add Gemini Analysis section if available
        if result.gemini_analysis and 'overall_summary' in result.gemini_analysis:
            st.subheader("AI Analysis")
            
            # Display overall summary
            st.markdown("### Overall Summary")
            st.write(result.gemini_analysis['overall_summary'])
            
            # Display comparison if available
            if 'comparison' in result.gemini_analysis:
                with st.expander("Comparative Analysis", expanded=True):
                    st.write(result.gemini_analysis['comparison'])
            
            # Display top locations with details
            st.markdown("### Top Recommended Locations")
            
            top_ids = result.gemini_analysis['top_locations']
            top_locations = [loc for loc in result.locations if loc.location_id in top_ids]
            
            for loc in top_locations:
                loc_id = str(loc.location_id)
                
                with st.expander(f"Location {loc.location_id}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Area:** {loc.area_hectares:.2f} hectares")
                        st.markdown(f"**Suitability Score:** {loc.suitability_score:.2f}")
                        st.markdown(f"**Coordinates:** {loc.latitude:.6f}, {loc.longitude:.6f}")
                        
                        if loc.address:
                            st.markdown(f"**Address:** {loc.address}")
                        
                        if loc.neighborhood:
                            st.markdown(f"**Neighborhood:** {loc.neighborhood}")
                        
                        if loc.city:
                            st.markdown(f"**City:** {loc.city}")
                    
                    with col2:
                        if loc_id in result.gemini_analysis['explanations']:
                            st.markdown("#### Analysis")
                            st.write(result.gemini_analysis['explanations'][loc_id])
                        
                        if loc_id in result.gemini_analysis['considerations']:
                            st.markdown("#### Development Considerations")
                            st.write(result.gemini_analysis['considerations'][loc_id])
                    
                    # Display nearby features if available
                    if loc.nearby_features:
                        st.markdown("#### Nearby Features")
                        
                        for layer, features in loc.nearby_features.items():
                            if features:
                                st.markdown(f"**{layer.title()}:**")
                                
                                try:
                                    # Create a small DataFrame for the features
                                    features_df = pd.DataFrame([
                                        {'Name': str(f.get('name', '')), 'Distance (m)': float(f.get('distance', 0))} 
                                        for f in features
                                    ])
                                    
                                    if not features_df.empty:
                                        features_df['Distance (m)'] = features_df['Distance (m)'].astype(float).round(0).astype(int)
                                        st.dataframe(features_df, use_container_width=True)
                                except Exception as e:
                                    # Fallback to simple text display
                                    for f in features[:5]:
                                        st.write(f"{str(f.get('name', 'Unknown'))} - {float(f.get('distance', 0)):.0f}m")
        
        # PDF Report section
        st.subheader("Generate Report")
        
        st.markdown(
            "Generate a comprehensive PDF report containing all analysis results, "
            "suitability maps, and location recommendations."
        )
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                # Create PDF service
                pdf_service = PDFReportService(result, boundary)
                
                # Create a temporary file for the PDF
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        temp_path = tmp.name
                        
                    # Generate the report
                    success = pdf_service.generate_report(temp_path)
                    
                    if success and os.path.exists(temp_path):
                        # Read the file
                        with open(temp_path, 'rb') as f:
                            pdf_data = f.read()
                        
                        # Create download button
                        st.success("PDF report generated successfully!")
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"suitability_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Failed to generate PDF report.")
                finally:
                    # Clean up the temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

        # Add chatbot UI
        add_simple_chatbot_ui(st, result, st.session_state.get('gemini_api_key'))

#############################################
# Helper Function
#############################################

def create_assistant(gemini_api_key):
    """Creates a simple conversational assistant."""
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    return GenerativeModel('gemini-1.5-pro')

def add_simple_chatbot_ui(st, analysis_result, gemini_api_key):
    """Add a chatbot UI to the results tab with comprehensive analysis context
    and GIS/urban planning expertise."""
    if analysis_result:
        st.subheader("Ask the GIS & Urban Planning Expert")
        
        expert_info = """
        This AI assistant combines analysis of your results with expertise in:
        - GIS methodology and spatial analysis
        - Urban planning and site development
        - Infrastructure and accessibility planning
        - Environmental and social considerations
        - Local context and regulations
        
        Ask about your results or seek professional insights about urban development.
        """
        
        with st.expander("About the Expert Assistant", expanded=False):
            st.markdown(expert_info)
        
        # Create assistant if not already in session state
        if 'assistant' not in st.session_state and gemini_api_key:
            st.session_state.assistant = create_assistant(gemini_api_key)
            
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Example questions to help users
        if len(st.session_state.chat_history) == 0:
            with st.expander("Example questions to ask", expanded=False):
                st.markdown("""
                - What additional analysis would improve the results?
                - What environmental factors should be considered for Location 1?
                - How would public transportation access impact these locations?
                - What zoning regulations might affect development at the top location?
                - What community engagement strategies would you recommend?
                - How does the proximity to schools impact the development potential?
                - What infrastructure challenges might these locations face?
                - Which location has the best development potential for affordable housing?
                - How would climate change factors impact these locations?
                - What GIS methodology would you recommend for refining this analysis?
                """)
        
        # Chat input
        query = st.chat_input("Ask a question about the analysis or urban planning:")
        
        if query and 'assistant' in st.session_state:
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
                
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_analysis_query(st.session_state.assistant, query, analysis_result)
                    st.markdown(response)
                    
            # Add to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        elif query:
            st.warning("Please provide a Gemini API key in the sidebar to enable the expert assistant.")
        
        # Add a button to clear chat history
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
#############################################
# Main Application
#############################################

def main():
    """Main application entry point."""
    # Load custom CSS
    load_custom_css()
    
    # App title
    st.title("Advanced Land Suitability Analysis Tool 🏙️")
    st.markdown(
        "Find optimal locations for development based on multiple spatial criteria, "
        "enhanced with AI-powered analysis."
    )
    
    # Render sidebar
    services = UIComponents.render_sidebar()
    db_service = services.get('db_service')
    gemini_service = services.get('gemini_service')
    
    # Create tabs
    analysis_tab, results_tab = st.tabs(["Analysis", "Results"])
    
    # Render each tab
    with analysis_tab:
        UIComponents.render_analysis_tab(db_service, gemini_service)
    
    with results_tab:
        UIComponents.render_results_tab()

if __name__ == "__main__":
    main()