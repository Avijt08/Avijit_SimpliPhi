"""
Test script to check available Gemini models
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No API key found!")
    exit(1)

genai.configure(api_key=api_key)

print("Testing available Gemini models...")

# List all available models
try:
    models = genai.list_models()
    print("\nAvailable models:")
    for model in models:
        print(f"- {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Supported methods: {model.supported_generation_methods}")
    
    # Test specific models
    test_models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-pro',
        'gemini-1.0-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro',
        'models/gemini-pro'
    ]
    
    print("\nTesting specific models:")
    working_models = []
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello'")
            print(f"✅ {model_name}: WORKING")
            working_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")
    
    print(f"\nWorking models: {working_models}")
    
except Exception as e:
    print(f"Error listing models: {e}")