import os
import sys
import subprocess
from pathlib import Path

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("Creating virtual environment...")
    
    # Create venv
    subprocess.run([sys.executable, "-m", "venv", "aura_env"], check=True)
    
    print("Virtual environment created successfully!")

def install_dependencies():
    """Install all required dependencies"""
    print("Installing dependencies...")
    
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "aura_env\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        pip_path = "aura_env/bin/pip"
    
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    
    print("Dependencies installed successfully!")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "xai_reports",
        "fingerprints", 
        "sentinel/received_models",
        "detector/training_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Setting up AURA environment...")
    
    try:
        create_virtual_environment()
        setup_directories()
        install_dependencies()
        
        print("\\n" + "="*50)
        print("AURA Setup Complete!")
        print("="*50)
        print("\\nTo activate the environment:")
        if os.name == 'nt':
            print("Windows: aura_env\\\\Scripts\\\\activate")
        else:
            print("Unix/Linux/MacOS: source aura_env/bin/activate")
        print("\\nTo run data generation: python scripts/generate_data.py")
        print("="*50)
        
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
