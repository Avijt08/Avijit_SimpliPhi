"""
Main entry point for the application
"""
import uvicorn
from pathlib import Path
import shutil
import os

def ensure_data_exists():
    """Ensure the data file exists in the correct location"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    property_data = data_dir / "Property_data.csv"
    if not property_data.exists():
        source_data = Path("../Property_data.csv")
        if source_data.exists():
            shutil.copy(source_data, property_data)
        else:
            raise FileNotFoundError("Property_data.csv not found")

if __name__ == "__main__":
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Check data file exists
    ensure_data_exists()
    
    # Start the FastAPI server
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )