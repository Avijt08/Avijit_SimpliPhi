"""
Enhanced LLM integration with OpenRouter and Google Gemini API support
"""
import os
import logging
import google.generativeai as genai
from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
from config import Config

logger = logging.getLogger(__name__)

class HybridLLMIntegration:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize Hybrid LLM integration with OpenRouter primary and Gemini fallback"""
        
        # OpenRouter configuration (primary)
        self.openrouter_api_key = Config.OPENROUTER_API_KEY
        self.openrouter_model = Config.OPENROUTER_MODEL
        self.openrouter_client = None
        
        # Gemini configuration (fallback)
        self.gemini_api_key = api_key or Config.GOOGLE_API_KEY
        self.gemini_model_name = model_name or Config.GEMINI_MODEL
        self.gemini_model = None
        
        # Initialize OpenRouter client
        if self.openrouter_api_key:
            try:
                self.openrouter_client = OpenAI(
                    base_url=Config.OPENROUTER_BASE_URL,
                    api_key=self.openrouter_api_key,
                )
                # Test the OpenRouter connection
                test_response = self.openrouter_client.chat.completions.create(
                    model=self.openrouter_model,
                    messages=[{"role": "user", "content": "Hello, test connection."}],
                    max_tokens=10
                )
                print(f"✅ OpenRouter API configured with model: {self.openrouter_model}")
                logger.info(f"OpenRouter API configured successfully with model: {self.openrouter_model}")
                self.primary_llm = "openrouter"
            except Exception as e:
                print(f"❌ Failed to initialize OpenRouter: {e}")
                logger.warning(f"Failed to initialize OpenRouter: {e}")
                self.openrouter_client = None
                self.primary_llm = "gemini"
        else:
            print("⚠️ No OpenRouter API key found. Trying Gemini...")
            self.primary_llm = "gemini"
        
        # Initialize Gemini as fallback (or primary if OpenRouter failed)
        if self.gemini_api_key and (not self.openrouter_client or self.primary_llm == "gemini"):
            genai.configure(api_key=self.gemini_api_key)
            
            # Try multiple model names in order of preference
            model_names_to_try = [
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro',
                'models/gemini-2.0-flash',
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                self.gemini_model_name
            ]
            
            for model_name in model_names_to_try:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    test_response = self.gemini_model.generate_content("Say 'test'")
                    print(f"✅ Gemini API configured with model: {model_name}")
                    logger.info(f"Gemini API configured successfully with model: {model_name}")
                    self.gemini_model_name = model_name
                    if not self.openrouter_client:
                        self.primary_llm = "gemini"
                    break
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini model {model_name}: {e}")
                    continue
        
        # Embedding model setup
        self.embedding_model = None
        print("🔧 Using hash-based embeddings for maximum compatibility")
        
        # Final status check
        if not self.openrouter_client and not self.gemini_model:
            logger.warning("No working LLM found. Responses will be simulated.")
            print("⚠️ No working LLM found. Responses will be simulated.")
            self.primary_llm = "fallback"
    
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
    
    def generate_response(self, query: str, context_properties: List[Dict[str, Any]]) -> str:
        """Generate a natural language response using property context"""
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
        
        # Try OpenRouter first
        if self.primary_llm == "openrouter" and self.openrouter_client:
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=self.openrouter_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error with OpenRouter: {str(e)}")
                print(f"❌ OpenRouter failed, trying Gemini fallback...")
                # Fall back to Gemini
        
        # Try Gemini
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Error with Gemini: {str(e)}")
                print(f"❌ Gemini failed, using fallback response...")
        
        # Fallback response
        return self._generate_fallback_response(query, context_properties)
    
    def _prepare_context(self, properties: List[Dict[str, Any]], max_properties: int = 5) -> str:
        """Prepare property context for the LLM with enhanced data handling"""
        context_parts = []
        
        for i, prop in enumerate(properties[:max_properties]):
            # Handle different data types and provide meaningful defaults
            property_type = prop.get('type') or prop.get('property_type', 'Property')
            price = prop.get('price')
            bedrooms = prop.get('bedrooms')
            bathrooms = prop.get('bathrooms')
            address = prop.get('address', 'Location not specified')
            crime_score = prop.get('crime_score_weight') or prop.get('crime_score')
            flood_risk = prop.get('flood_risk')
            
            context_part = f"""
            Property {i+1}:
            - Type: {property_type}
            - Price: £{price:,.2f} ({self._price_category(price)})
            - Bedrooms: {bedrooms if bedrooms is not None else 'Not specified'}
            - Bathrooms: {bathrooms if bathrooms is not None else 'Not specified'}
            - Location: {address}
            """
            
            # Add crime score if available
            if crime_score is not None:
                try:
                    crime_val = float(crime_score)
                    crime_desc = "Low" if crime_val < 0.4 else "Medium" if crime_val < 0.7 else "High"
                    context_part += f"- Crime Level: {crime_desc} (score: {crime_val:.2f})\n"
                except (ValueError, TypeError):
                    context_part += f"- Crime Level: Not available\n"
            else:
                context_part += f"- Crime Level: Not available\n"
            
            # Add flood risk if available
            if flood_risk and flood_risk not in ['', 'N/A', 'Unknown']:
                context_part += f"- Flood Risk: {flood_risk}\n"
            else:
                context_part += f"- Flood Risk: Not specified\n"
            
            # Add description if available
            if prop.get('property_type_full_description'):
                context_part += f"- Description: {prop.get('property_type_full_description')}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _price_category(self, price) -> str:
        """Categorize price for better context"""
        if price is None:
            return "Price not specified"
        
        try:
            price = float(price)
            if price == 0:
                return "Price not available"
            elif price < 200000:
                return "Budget-friendly"
            elif price < 400000:
                return "Mid-range"
            elif price < 600000:
                return "Premium"
            else:
                return "Luxury"
        except (ValueError, TypeError):
            return "Price not available"
    
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

# Backward compatibility alias
GeminiLLMIntegration = HybridLLMIntegration