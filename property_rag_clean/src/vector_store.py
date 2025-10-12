"""
Enhanced Property Vector Store with simple in-memory storage and filtering
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from llm_integration import GeminiLLMIntegration

class PropertyVectorStore:
    def __init__(self, data_file: str = None):
        self.properties = []
        self.embeddings = []
        self.llm_integration = GeminiLLMIntegration()
        
        print("🗄️ Using optimized in-memory vector storage")
        
        if data_file and os.path.exists(data_file):
            self.load_data(data_file)
    
    def load_data(self, data_file: str):
        """Load property data from CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            self.properties = df.to_dict('records')
            print(f"Loaded {len(self.properties)} properties")
        except ImportError:
            print("⚠️ Pandas not available, loading with basic CSV reader")
            # Fallback CSV reader
            import csv
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.properties = list(reader)
            print(f"Loaded {len(self.properties)} properties")
    
    def add_property(self, property_data: Dict[str, Any], embedding: Optional[List[float]] = None):
        """Add a property with its embedding"""
        self.properties.append(property_data)
        if embedding:
            self.embeddings.append(embedding)
    
    def generate_embeddings(self, force_regenerate: bool = False):
        """Generate embeddings for all properties with simple hash-based method"""
        if self.embeddings and not force_regenerate:
            print("Embeddings already exist. Use force_regenerate=True to regenerate.")
            return
        
        print(f"Generating embeddings for {len(self.properties)} properties...")
        self.embeddings = []
        
        for i, property_data in enumerate(self.properties):
            searchable_text = self._create_searchable_text(property_data)
            embedding = self.llm_integration.generate_embeddings([searchable_text])
            
            if embedding and len(embedding) > 0:
                self.embeddings.append(embedding[0])
            else:
                # Fallback: create a dummy embedding
                self.embeddings.append([0.0] * 384)
            
            if (i + 1) % 100 == 0:
                print(f"Generated embeddings for {i + 1}/{len(self.properties)} properties")
        
        print("Embedding generation complete!")
    
    def _create_searchable_text(self, property_data: Dict[str, Any]) -> str:
        """Create searchable text from property data"""
        text_parts = []
        
        # Add key fields to searchable text
        for key, value in property_data.items():
            if value is not None and str(value).strip():
                if key.lower() in ['type', 'location', 'city', 'area', 'description']:
                    text_parts.append(f"{key}: {value}")
                else:
                    text_parts.append(str(value))
        
        return " ".join(text_parts)
        
    def _matches_price_filter(self, metadata: Dict[str, Any], price_min: float = None, price_max: float = None) -> bool:
        """Check if property matches price filter"""
        if price_min is None and price_max is None:
            return True
        
        try:
            price = float(metadata.get('price', 0))
            if price_min is not None and price < price_min:
                return False
            if price_max is not None and price > price_max:
                return False
            return True
        except (ValueError, TypeError):
            return True  # Include properties with invalid price data
        
    def get_property_count(self) -> int:
        """Get total number of properties"""
        return len(self.properties)
    
    def get_embedding_count(self) -> int:
        """Get number of embeddings available"""
        if self.use_chromadb and self.collection:
            try:
                return self.collection.count()
            except:
                return 0
        return len(self.embeddings)
    
    def clear_embeddings(self):
        """Clear all embeddings"""
        if self.use_chromadb and self.collection:
            try:
                # Delete the collection and recreate it
                self.client.delete_collection("property_embeddings")
                self.collection = self.client.get_or_create_collection(
                    name="property_embeddings",
                    metadata={"description": "Property search embeddings"}
                )
                print("ChromaDB embeddings cleared")
            except Exception as e:
                print(f"Error clearing ChromaDB: {e}")
        
        self.embeddings = []
        print("In-memory embeddings cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_properties': self.get_property_count(),
            'total_embeddings': self.get_embedding_count(),
            'using_chromadb': False,
            'chromadb_available': False
        }
    
    def semantic_search(self, query: str, top_k: int = 5, price_min: float = None, price_max: float = None) -> List[Dict[str, Any]]:
        """Perform semantic search with optional price filtering"""
        if not self.embeddings:
            print("No embeddings available. Performing text-based search...")
            return self.text_search(query, top_k, price_min, price_max)
        
        # Generate embedding for query
        query_embedding = self.llm_integration.generate_embeddings([query])
        
        if not query_embedding or len(query_embedding) == 0:
            print("Could not generate query embedding. Falling back to text search...")
            return self.text_search(query, top_k, price_min, price_max)
        
        query_vec = np.array(query_embedding[0])
        scores = []
        
        # Calculate similarity scores
        for i, embedding in enumerate(self.embeddings):
            if self._matches_price_filter(self.properties[i], price_min, price_max):
                try:
                    embedding_vec = np.array(embedding)
                    # Cosine similarity
                    similarity = np.dot(query_vec, embedding_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(embedding_vec)
                    )
                    scores.append((similarity, i))
                except Exception as e:
                    # Skip invalid embeddings
                    continue
        
        # Sort by similarity and return top results
        scores.sort(reverse=True)
        results = []
        
        for score, idx in scores[:top_k]:
            results.append({
                'content': self._create_searchable_text(self.properties[idx]),
                'metadata': self.properties[idx],
                'score': float(score)
            })
        
        return results
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            if embedding:
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for i, (prop_idx, similarity) in enumerate(similarities[:top_k]):
            property_data = self.properties[prop_idx].copy()
            property_data['similarity_score'] = similarity
            results.append(property_data)
        
        return results
    
    def text_search(self, query: str, top_k: int = 5, price_min: float = None, price_max: float = None) -> List[Dict[str, Any]]:
        """Enhanced text-based search with price filtering and better scoring"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        for i, property_data in enumerate(self.properties):
            # Apply price filter first
            if not self._matches_price_filter(property_data, price_min, price_max):
                continue
                
            searchable_text = self._create_searchable_text(property_data).lower()
            
            # Calculate relevance score
            score = 0
            
            # Exact phrase match (highest weight)
            if query_lower in searchable_text:
                score += 10
            
            # Word matches
            text_words = set(searchable_text.split())
            word_matches = len(query_words.intersection(text_words))
            score += word_matches * 2
            
            # Partial word matches
            for query_word in query_words:
                for text_word in text_words:
                    if query_word in text_word or text_word in query_word:
                        score += 0.5
            
            # Property type matches (higher weight)
            property_type = str(property_data.get('type', '')).lower()
            for word in query_words:
                if word in property_type:
                    score += 3
            
            # Price range mentions
            price_keywords = ['cheap', 'affordable', 'expensive', 'luxury', 'budget']
            for keyword in price_keywords:
                if keyword in query_lower:
                    price = float(property_data.get('price', 0))
                    if keyword in ['cheap', 'affordable', 'budget'] and price < 2000:
                        score += 2
                    elif keyword in ['expensive', 'luxury'] and price > 3000:
                        score += 2
            
            if score > 0:
                scores.append((score, i))
        
        # Sort by score and return top results
        scores.sort(reverse=True)
        results = []
        
        for score, idx in scores[:top_k]:
            results.append({
                'content': self._create_searchable_text(self.properties[idx]),
                'metadata': self.properties[idx],
                'score': float(score)
            })
        
        return results
        query_lower = query.lower()
        scored_properties = []
        
        for prop in self.properties:
            score = 0
            searchable_text = self._create_searchable_text(prop).lower()
            
            # Simple keyword matching
            query_words = query_lower.split()
            for word in query_words:
                if word in searchable_text:
                    score += 1
            
            if score > 0:
                prop_copy = prop.copy()
                prop_copy['similarity_score'] = score / len(query_words)
                scored_properties.append(prop_copy)
        
        # Sort by score
        scored_properties.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return scored_properties[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            return dot_product / (norm_vec1 * norm_vec2)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file"""
        try:
            data = {
                'properties': self.properties,
                'embeddings': self.embeddings
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str)
            print(f"Embeddings saved to {filepath}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.properties = data['properties']
            self.embeddings = data['embeddings']
            print(f"Embeddings loaded from {filepath}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
    
    def get_property_count(self) -> int:
        """Get number of properties in the store"""
        return len(self.properties)
    
    def get_embedding_count(self) -> int:
        """Get number of embeddings generated"""
        return len(self.embeddings)