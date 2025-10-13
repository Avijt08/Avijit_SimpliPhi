"""
Test script for OpenRouter integration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'property_rag_clean', 'src'))

from llm_integration import HybridLLMIntegration

def test_openrouter():
    """Test the OpenRouter integration"""
    print("🧪 Testing OpenRouter Integration...")
    
    # Initialize the LLM integration
    llm = HybridLLMIntegration()
    
    # Test query
    test_query = "Find me a 3-bedroom house under £500,000"
    test_properties = [
        {
            'type': 'House',
            'price': 450000,
            'bedrooms': 3,
            'bathrooms': 2,
            'address': '123 Oak Street, London',
            'crime_score_weight': 0.3,
            'flood_risk': 'Low'
        },
        {
            'type': 'Apartment',
            'price': 320000,
            'bedrooms': 2,
            'bathrooms': 1,
            'address': '45 Pine Avenue, Manchester',
            'crime_score_weight': 0.2,
            'flood_risk': 'None'
        }
    ]
    
    print(f"Query: {test_query}")
    print("Testing LLM response generation...")
    
    try:
        response = llm.generate_response(test_query, test_properties)
        print("\n✅ LLM Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Test embeddings
        print("\n🔧 Testing embedding generation...")
        embeddings = llm.generate_embeddings([test_query])
        print(f"✅ Generated embedding with dimension: {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openrouter()
    if success:
        print("\n🎉 OpenRouter integration test completed successfully!")
    else:
        print("\n💥 OpenRouter integration test failed!")