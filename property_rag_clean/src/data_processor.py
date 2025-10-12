"""
Enhanced data processor for property data with RAG-optimized descriptions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load and clean the property data"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} properties from {self.data_path}")
            
            # Clean column names
            self.df.columns = self.df.columns.str.lower()
            
            # Data cleaning and preprocessing
            self._clean_data()
            logger.info("Data cleaning completed")
            
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        # Create a copy to avoid warnings
        self.df = self.df.copy()
        
        # Handle missing values intelligently
        self.df['property_type_full_description'] = self.df['property_type_full_description'].fillna(self.df['type'])
        self.df['flood_risk'] = self.df['flood_risk'].fillna('None')
        self.df['address'] = self.df['address'].fillna('Unknown Location')
        
        # Handle crime score - fill with median for each area if possible
        if 'crime_score_weight' in self.df.columns:
            self.df['crime_score_weight'] = self.df['crime_score_weight'].fillna(
                self.df.groupby('address')['crime_score_weight'].transform('median')
            )
            # Fill remaining NaN with overall median
            self.df['crime_score_weight'] = self.df['crime_score_weight'].fillna(
                self.df['crime_score_weight'].median()
            )
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['price', 'bedrooms', 'bathrooms', 'crime_score_weight']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
    
    def create_rag_property_document(self, row) -> str:
        """Create a comprehensive, RAG-optimized description for a property"""
        document_parts = []
        
        # Basic property information
        property_type = str(row.get('type', 'Property'))
        price = row.get('price', 0)
        bedrooms = row.get('bedrooms', 0)
        bathrooms = row.get('bathrooms', 0)
        
        # Main description
        document_parts.append(f"This is a {property_type.lower()} priced at £{price:,.2f}")
        
        # Room information
        if bedrooms > 0:
            document_parts.append(f"featuring {bedrooms} bedrooms and {bathrooms} bathrooms")
        elif property_type.lower() == 'studio':
            document_parts.append("featuring a studio layout with open living space")
        else:
            document_parts.append(f"with {bathrooms} bathrooms")
        
        # Location
        if pd.notna(row.get('address')) and str(row.get('address')) != 'Unknown Location':
            document_parts.append(f"located in {row['address']}")
        
        # Property description
        if pd.notna(row.get('property_type_full_description')):
            desc = str(row['property_type_full_description'])
            if desc and desc != str(row.get('type', '')):
                document_parts.append(f"described as: {desc}")
        
        # Safety and risk information
        if pd.notna(row.get('crime_score_weight')):
            crime_score = float(row['crime_score_weight'])
            if crime_score < 3:
                document_parts.append("in a very safe low-crime area")
            elif crime_score < 5:
                document_parts.append("in a safe area with low crime rates")
            elif crime_score < 7:
                document_parts.append("in an area with moderate crime levels")
            else:
                document_parts.append("in an area with higher crime rates")
        
        # Flood risk
        if pd.notna(row.get('flood_risk')):
            flood_risk = str(row['flood_risk']).lower()
            if flood_risk in ['low', 'very low']:
                document_parts.append("with minimal flood risk")
            elif flood_risk == 'medium':
                document_parts.append("with moderate flood risk")
            elif flood_risk == 'high':
                document_parts.append("with higher flood risk")
            elif flood_risk == 'none':
                document_parts.append("with no flood risk")
        
        # New home indicator
        if row.get('is_new_home') == 'TRUE':
            document_parts.append("and is a newly built property")
        
        # Price category for search optimization
        if price < 1000:
            document_parts.append("making it an affordable budget option")
        elif price < 2000:
            document_parts.append("offering good value for money")
        elif price < 3500:
            document_parts.append("in the mid-range price category")
        else:
            document_parts.append("positioned as a premium property")
        
        return ". ".join(document_parts) + "."
    
    def prepare_for_rag(self) -> Dict[str, List]:
        """Prepare data for RAG system with enhanced documents and metadata"""
        if self.df is None:
            self.load_data()
        
        # Create enhanced documents for each property
        documents = []
        metadatas = []
        
        for idx, row in self.df.iterrows():
            # Create comprehensive document
            document = self.create_rag_property_document(row)
            documents.append(document)
            
            # Create metadata with proper type handling
            def safe_int(value, default=0):
                if pd.isna(value):
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_float(value, default=0.0):
                if pd.isna(value):
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Create the property dictionary with safe conversions
            property_dict = {
                'id': str(idx),
                'type': str(row.get('type', 'Unknown')),
                'price': safe_float(row.get('price', 0)),
                'bedrooms': safe_int(row.get('bedrooms', 0)),
                'bathrooms': safe_int(row.get('bathrooms', 0)),
                'address': str(row.get('address', 'Unknown')),
                'description': document,
                'original_description': str(row.get('property_type_full_description', '')),
            }
            
            # Add optional fields if available
            if 'crime_score_weight' in row and pd.notna(row['crime_score_weight']):
                property_dict['crime_score'] = safe_float(row['crime_score_weight'])
            
            if 'flood_risk' in row and pd.notna(row['flood_risk']):
                property_dict['flood_risk'] = str(row['flood_risk'])
            
            if 'latitude' in row and pd.notna(row['latitude']):
                property_dict['latitude'] = safe_float(row['latitude'])
                
            if 'longitude' in row and pd.notna(row['longitude']):
                property_dict['longitude'] = safe_float(row['longitude'])
            
            if 'listing_update_date' in row and pd.notna(row['listing_update_date']):
                property_dict['listing_date'] = str(row['listing_update_date'])
            
            if 'is_new_home' in row:
                property_dict['is_new_home'] = str(row.get('is_new_home', 'FALSE'))
            
            metadatas.append(property_dict)
        
        logger.info(f"Prepared {len(documents)} properties for RAG system")
        
        return {
            'documents': documents,
            'metadatas': metadatas
        }
    
    def get_property_summaries(self):
        """Legacy method for backward compatibility"""
        rag_data = self.prepare_for_rag()
        return rag_data['documents']
    
    def get_metadata(self):
        """Legacy method for backward compatibility"""
        rag_data = self.prepare_for_rag()
        return rag_data['metadatas']