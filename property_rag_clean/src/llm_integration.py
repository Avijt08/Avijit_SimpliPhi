"""
Enhanced Google Gemini API integration with simple embedding fallback
"""
import os
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
from config import Config

logger = logging.getLogger(__name__)

class GeminiLLMIntegration:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize Gemini LLM integration with fallback support"""
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL
        
        # Simple embedding model (no external dependencies)
        self.embedding_model = None
        print("🔧 Using hash-based embeddings for maximum compatibility")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
            # Try multiple model names in order of preference (updated with working models)
            model_names_to_try = [
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro',
                'models/gemini-2.0-flash',
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                self.model_name  # User-specified model as fallback
            ]
            
            self.model = None
            for model_name in model_names_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    test_response = self.model.generate_content("Say 'test'")
                    logger.info(f"Gemini API configured successfully with model: {model_name}")
                    print(f"✅ Gemini API configured with model: {model_name}")
                    self.model_name = model_name
                    break
                except Exception as e:
                    logger.warning(f"Failed to initialize model {model_name}: {e}")
                    print(f"❌ Failed to initialize model {model_name}: {e}")
                    continue
            
            if self.model is None:
                logger.warning("No working Gemini model found. LLM responses will be simulated.")
                print("⚠️ No working Gemini model found. LLM responses will be simulated.")
                
        else:
            logger.warning("No Google API key found. LLM responses will be simulated.")
            print("⚠️ No Google API key found. LLM responses will be simulated.")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using simple hash-based method for compatibility"""
        # Convert single string to list if needed
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create a simple but effective embedding using hash-based method
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to a simple embedding vector"""
        # Simple hash-based embedding for fallback
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert to numbers and normalize to create a 384-dimensional vector
        # (matching Sentence Transformers dimension)
        embedding = []
        for i in range(0, min(len(text_hash), 32), 2):
            try:
                val = int(text_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
                embedding.append(val)
            except:
                embedding.append(0.5)
        
        # Pad or truncate to 384 dimensions
        while len(embedding) < 384:
            embedding.extend(embedding[:min(384-len(embedding), len(embedding))])
        
        return embedding[:384]
        # This is a very basic implementation - in practice, use proper embedding models
        words = text.lower().split()[:768]  # Limit to 768 dimensions
        embedding = [0.0] * 768
        
        for i, word in enumerate(words):
            if i < 768:
                # Simple hash-based embedding
                embedding[i] = (hash(word) % 1000) / 1000.0
        
        return embedding
    
    def generate_response(self, query: str, context_properties: List[Dict[str, Any]]) -> str:
        """Generate a natural language response using property context"""
        if not self.api_key:
            return self._generate_fallback_response(query, context_properties)
        
        try:
            # Prepare context from properties
            context = self._prepare_context(context_properties)
            
            prompt = f"""
            You are a helpful property search assistant. Based on the following property data, 
            provide a helpful and informative response to the user's query.
            
            User Query: {query}
            
            Relevant Properties:
            {context}
            
            Please provide a helpful response that:
            1. Addresses the user's specific query
            2. Highlights the most relevant properties from the search results
            3. Provides useful insights about price ranges, locations, or features
            4. Is conversational and friendly
            5. Includes specific property details when relevant
            
            Response:
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return self._generate_fallback_response(query, context_properties)
    
    def _prepare_context(self, properties: List[Dict[str, Any]], max_properties: int = 5) -> str:
        """Prepare property context for the LLM"""
        context_parts = []
        
        for i, prop in enumerate(properties[:max_properties]):
            context_part = f"""
            Property {i+1}:
            - Type: {prop.get('type', 'Unknown')}
            - Price: £{prop.get('price', 0):,.2f}
            - Bedrooms: {prop.get('bedrooms', 'N/A')}
            - Bathrooms: {prop.get('bathrooms', 'N/A')}
            - Location: {prop.get('address', 'Unknown')}
            - Crime Score: {prop.get('crime_score_weight', 'N/A')}
            - Flood Risk: {prop.get('flood_risk', 'N/A')}
            """
            
            if prop.get('property_type_full_description'):
                context_part += f"- Description: {prop.get('property_type_full_description')}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, query: str, properties: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when API is not available"""
        if not properties:
            return f"I couldn't find any properties matching your query '{query}'. Try adjusting your search criteria or looking for different property types or locations."
        
        total_properties = len(properties)
        avg_price = sum(p.get('price', 0) for p in properties) / len(properties)
        
        # Get unique property types and locations
        property_types = list(set(p.get('type', 'Unknown') for p in properties))
        locations = list(set(p.get('address', 'Unknown') for p in properties if p.get('address') != 'Unknown'))
        
        response = f"I found {total_properties} properties matching your search for '{query}'.\n\n"
        
        response += f"**Summary:**\n"
        response += f"- Average price: £{avg_price:,.2f}\n"
        response += f"- Property types: {', '.join(property_types[:3])}\n"
        response += f"- Locations include: {', '.join(locations[:3])}\n\n"
        
        if total_properties > 0:
            top_property = properties[0]
            response += f"**Featured Property:**\n"
            response += f"A {top_property.get('type', 'property')} in {top_property.get('address', 'unknown location')} "
            response += f"for £{top_property.get('price', 0):,.2f} with {top_property.get('bedrooms', 'unknown')} bedrooms "
            response += f"and {top_property.get('bathrooms', 'unknown')} bathrooms.\n\n"
        
        response += "Browse through the search results below to see all matching properties with detailed information."
        
        return response
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the user's query to extract search intent and parameters"""
        query_lower = query.lower()
        
        # Extract potential filters from natural language
        filters = {}
        
        # Price analysis
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable', 'under']):
            filters['price_preference'] = 'low'
        elif any(word in query_lower for word in ['luxury', 'expensive', 'premium', 'high-end']):
            filters['price_preference'] = 'high'
        
        # Property type analysis
        if any(word in query_lower for word in ['apartment', 'flat']):
            filters['property_type'] = 'apartment'
        elif any(word in query_lower for word in ['house', 'home']):
            filters['property_type'] = 'house'
        elif 'studio' in query_lower:
            filters['property_type'] = 'studio'
        
        # Location analysis
        location_keywords = ['in', 'near', 'around', 'area', 'location']
        for keyword in location_keywords:
            if keyword in query_lower:
                # Extract potential location after the keyword
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    potential_location = parts[1].strip().split()[0] if parts[1].strip() else None
                    if potential_location:
                        filters['location_hint'] = potential_location
                break
        
        # Feature analysis
        if any(word in query_lower for word in ['garden', 'outdoor', 'yard']):
            filters['features'] = filters.get('features', []) + ['garden']
        if any(word in query_lower for word in ['safe', 'secure', 'low crime']):
            filters['features'] = filters.get('features', []) + ['safe_area']
        
        return {
            'original_query': query,
            'extracted_filters': filters,
            'query_type': self._classify_query_type(query_lower)
        }
    
    def _classify_query_type(self, query_lower: str) -> str:
        """Classify the type of query"""
        if any(word in query_lower for word in ['what', 'how', 'why', 'explain']):
            return 'informational'
        elif any(word in query_lower for word in ['find', 'search', 'show', 'need', 'want', 'looking']):
            return 'search'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        else:
            return 'general'