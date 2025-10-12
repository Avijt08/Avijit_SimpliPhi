from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd
import traceback

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_processor import DataProcessor
from vector_store import PropertyVectorStore
from llm_integration import GeminiLLMIntegration

app = FastAPI(title="Property RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = None
llm_integration = None
data_processor = None

class PropertyQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5
    use_ai_response: Optional[bool] = True
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    property_type: Optional[str] = None

class RAGResponse(BaseModel):
    query: str
    properties: List[Dict[str, Any]]
    ai_response: Optional[str] = None
    query_intent: Optional[str] = None
    total_found: int

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global vector_store, llm_integration, data_processor
    
    try:
        print("Initializing Property RAG system...")
        
        # Initialize components
        config = Config()
        llm_integration = GeminiLLMIntegration()
        
        # Load data
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Property_data.csv")
        data_processor = DataProcessor(data_file)
        
        if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            vector_store = PropertyVectorStore(data_file)
            
            # Process data for RAG
            data_processor.load_data()
            processed_data = data_processor.prepare_for_rag()
            print(f"Processed {len(processed_data.get('documents', []))} properties for RAG")
            
            # Skip embeddings generation on startup for faster initialization
            print("Skipping embeddings generation on startup. Will generate on first query if needed.")
            
            print("Property RAG system initialized successfully!")
        else:
            print(f"Warning: Data file not found at {data_file}")
            vector_store = PropertyVectorStore()
            
    except Exception as e:
        print(f"Error during startup: {e}")
        traceback.print_exc()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Property RAG API",
        "version": "1.0.0",
        "status": "running",
        "properties_loaded": vector_store.get_property_count() if vector_store else 0,
        "embeddings_generated": vector_store.get_embedding_count() if vector_store else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "llm_integration_available": llm_integration is not None,
        "properties_count": vector_store.get_property_count() if vector_store else 0
    }

@app.post("/query", response_model=RAGResponse)
async def query_properties(query_request: PropertyQuery):
    """Enhanced query endpoint with RAG capabilities and filtering"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        query = query_request.query
        max_results = query_request.max_results
        use_ai_response = query_request.use_ai_response
        price_min = query_request.price_min
        price_max = query_request.price_max
        property_type = query_request.property_type
        
        print(f"Processing query: {query}")
        if price_min or price_max:
            print(f"Price filter: £{price_min or 0} - £{price_max or 'unlimited'}")
        
        # Analyze query intent if LLM is available
        query_intent = None
        if llm_integration:
            try:
                intent_result = llm_integration.analyze_query_intent(query)
                # Convert dict to string for the response model
                query_intent = str(intent_result.get('query_type', 'general')) if isinstance(intent_result, dict) else str(intent_result)
            except Exception as e:
                print(f"Error analyzing query intent: {e}")
        
        # Perform enhanced semantic search with filtering
        properties = vector_store.semantic_search(
            query, 
            max_results, 
            price_min=price_min, 
            price_max=price_max
        )
        
        # Apply property type filter if specified
        if property_type:
            properties = [
                prop for prop in properties 
                if str(prop.get('metadata', {}).get('type', '')).lower() == property_type.lower()
            ][:max_results]
        
        # Generate AI response if requested and LLM is available
        ai_response = None
        if use_ai_response and llm_integration and properties:
            try:
                ai_response = llm_integration.generate_response(query, properties)
            except Exception as e:
                print(f"Error generating AI response: {e}")
                ai_response = "I found some properties matching your query, but couldn't generate a detailed response at the moment."
        
        return RAGResponse(
            query=query,
            properties=properties,
            ai_response=ai_response,
            query_intent=query_intent,
            total_found=len(properties)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/filter")
async def filter_properties(
    price_min: Optional[float] = Query(None, description="Minimum price"),
    price_max: Optional[float] = Query(None, description="Maximum price"),
    property_type: Optional[str] = Query(None, description="Property type"),
    bedrooms: Optional[int] = Query(None, description="Number of bedrooms"),
    limit: int = Query(20, description="Maximum number of results")
):
    """Filter properties by various criteria"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        # Get all properties and apply filters
        filtered_properties = []
        for prop in vector_store.properties:
            # Price filter
            if price_min is not None and prop.get('price', 0) < price_min:
                continue
            if price_max is not None and prop.get('price', float('inf')) > price_max:
                continue
            
            # Property type filter
            if property_type and str(prop.get('type', '')).lower() != property_type.lower():
                continue
            
            # Bedrooms filter
            if bedrooms is not None and prop.get('bedrooms', 0) != bedrooms:
                continue
            
            filtered_properties.append(prop)
            
            if len(filtered_properties) >= limit:
                break
        
        return {
            "filters": {
                "price_min": price_min,
                "price_max": price_max,
                "property_type": property_type,
                "bedrooms": bedrooms
            },
            "total_found": len(filtered_properties),
            "properties": filtered_properties
        }
        
    except Exception as e:
        print(f"Error filtering properties: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics and analytics"""
    try:
        stats = {
            "system": {
                "vector_store_initialized": vector_store is not None,
                "llm_integration_available": llm_integration is not None,
                "properties_count": vector_store.get_property_count() if vector_store else 0,
                "embeddings_count": vector_store.get_embedding_count() if vector_store else 0,
            }
        }
        
        if vector_store:
            # Add vector store stats
            stats["vector_store"] = vector_store.get_stats()
            
            # Property analytics
            if vector_store.properties:
                import pandas as pd
                df = pd.DataFrame(vector_store.properties)
                
                # Price statistics
                if 'price' in df.columns:
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    stats["analytics"] = {
                        "price_stats": {
                            "min": float(df['price'].min()),
                            "max": float(df['price'].max()),
                            "mean": float(df['price'].mean()),
                            "median": float(df['price'].median())
                        },
                        "property_types": df['type'].value_counts().to_dict() if 'type' in df.columns else {},
                        "bedroom_distribution": df['bedrooms'].value_counts().to_dict() if 'bedrooms' in df.columns else {}
                    }
        
        return stats
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"error": str(e)}

@app.get("/search")
async def search_properties(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, description="Maximum number of results")
):
    """Simple search endpoint for backward compatibility"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        properties = vector_store.semantic_search(q, limit)
        
        return {
            "query": q,
            "results": properties,
            "total": len(properties)
        }
        
    except Exception as e:
        print(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/properties")
async def get_all_properties(
    skip: int = Query(0, description="Number of properties to skip"),
    limit: int = Query(10, description="Maximum number of properties to return")
):
    """Get all properties with pagination"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        all_properties = vector_store.properties
        total = len(all_properties)
        
        # Apply pagination
        paginated_properties = all_properties[skip:skip + limit]
        
        return {
            "properties": paginated_properties,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        print(f"Error getting properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate-embeddings")
async def regenerate_embeddings():
    """Regenerate embeddings for all properties"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        vector_store.generate_embeddings(force_regenerate=True)
        
        return {
            "message": "Embeddings regenerated successfully",
            "properties_count": vector_store.get_property_count(),
            "embeddings_count": vector_store.get_embedding_count()
        }
        
    except Exception as e:
        print(f"Error regenerating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        return {
            "total_properties": vector_store.get_property_count(),
            "embeddings_generated": vector_store.get_embedding_count(),
            "llm_available": llm_integration is not None,
            "api_status": "running"
        }
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)