import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
import urllib.request
import zipfile
import tarfile

# Project metadata
PROJECT_NAME = "sentiment-analysis-ai"
VERSION = "1.0.0"
DESCRIPTION = "AI-powered sentiment analysis with ML and Deep Learning models"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"

# Required packages
REQUIRED_PACKAGES = [
    # Core Data Science
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    
    # NLP
    "nltk>=3.8.0",
    "textblob>=0.17.0",
    "gensim>=4.3.0",
    
    # Deep Learning
    "tensorflow>=2.13.0",
    "keras>=2.13.0",
    
    # Web Framework
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "wordcloud>=1.9.0",
    
    # Utilities
    "requests>=2.31.0",
    "tqdm>=4.65.0",
    "joblib>=1.3.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    
    # Jupyter (optional)
    "ipykernel>=6.25.0",
    "jupyter>=1.0.0",
]

# NLTK data packages required
NLTK_DATA_PACKAGES = [
    'punkt',
    'stopwords',
    'wordnet',
    'vader_lexicon',
    'averaged_perceptron_tagger',
    'omw-1.4'
]

class ProjectSetup:
    """Handles all project setup tasks"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.venv_dir = self.base_dir / "venv"
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n🔍 Checking Python version...")
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} detected")
            print("   Required: Python 3.8 or higher")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def create_directory_structure(self):
        """Create all necessary directories"""
        self.print_header("CREATING DIRECTORY STRUCTURE")
        
        directories = [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.base_dir / "src",
            self.base_dir / "app",
            self.base_dir / "notebooks",
            self.base_dir / "tests",
            self.base_dir / "docs"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"  📁 {directory.name}/")
        
        print("✅ All directories created")
    
    def setup_virtual_environment(self, force=False):
        """Create and setup virtual environment"""
        self.print_header("SETTING UP VIRTUAL ENVIRONMENT")
        
        if self.venv_dir.exists() and not force:
            print("⚠️  Virtual environment already exists")
            response = input("   Recreate? (y/n): ").lower()
            if response != 'y':
                print("   Skipping...")
                return True
        
        # Remove existing venv if force
        if self.venv_dir.exists() and force:
            import shutil
            shutil.rmtree(self.venv_dir)
        
        print("📦 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_dir)])
            print("✅ Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self):
        """Get Python executable path in virtual environment"""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
    
    def get_venv_pip(self):
        """Get pip executable path in virtual environment"""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
    
    def install_dependencies(self, upgrade=False):
        """Install all required packages"""
        self.print_header("INSTALLING DEPENDENCIES")
        
        pip = self.get_venv_pip()
        
        if not pip.exists():
            print("❌ Virtual environment not found. Run setup first.")
            return False
        
        # Upgrade pip first
        print("⬆️  Upgrading pip...")
        subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
        
        # Install packages
        print(f"\n📥 Installing {len(REQUIRED_PACKAGES)} packages...")
        print("   This may take 5-10 minutes...\n")
        
        for package in REQUIRED_PACKAGES:
            print(f"   Installing {package}...")
            try:
                cmd = [str(pip), "install", package]
                if upgrade:
                    cmd.append("--upgrade")
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"   ✅ {package.split('>=')[0]}")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  Failed to install {package}")
        
        print("\n✅ Dependencies installation complete")
        return True
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        self.print_header("DOWNLOADING NLTK DATA")
        
        try:
            import nltk
            
            for package in NLTK_DATA_PACKAGES:
                print(f"   📥 Downloading {package}...")
                nltk.download(package, quiet=True)
                print(f"   ✅ {package}")
            
            print("\n✅ NLTK data download complete")
            return True
        except ImportError:
            print("❌ NLTK not installed. Install dependencies first.")
            return False
    
    def create_env_file(self):
        """Create .env file for environment variables"""
        env_file = self.base_dir / ".env"
        
        if env_file.exists():
            print("⚠️  .env file already exists")
            return
        
        env_content = """# Sentiment Analysis Configuration
FLASK_APP=app/app.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-change-this-in-production

# Model Settings
DEFAULT_MODEL=best
MAX_TEXT_LENGTH=1000
BATCH_SIZE=32

# API Settings (for future use)
API_RATE_LIMIT=100
API_TIMEOUT=30

# Paths
DATA_PATH=./data
MODELS_PATH=./models
LOGS_PATH=./logs
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env file")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore = self.base_dir / ".gitignore"
        
        if gitignore.exists():
            return
        
        content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
models/*.pkl
models/*.h5
data/*.csv
data/*.json
logs/
predictions/
*.log

# OS
.DS_Store
Thumbs.db
"""
        
        with open(gitignore, 'w') as f:
            f.write(content)
        
        print("✅ Created .gitignore")
    
    def download_sample_data(self):
        """Download sample datasets from online sources"""
        self.print_header("DOWNLOADING SAMPLE DATA")
        
        datasets = {
            'twitter_sample.csv': 'https://raw.githubusercontent.com/your-repo/sample-data/main/twitter_sample.csv',
            # Add more URLs as needed
        }
        
        for filename, url in datasets.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"   ⏭️  {filename} already exists")
                continue
            
            try:
                print(f"   📥 Downloading {filename}...")
                urllib.request.urlretrieve(url, str(filepath))
                print(f"   ✅ {filename}")
            except Exception as e:
                print(f"   ⚠️  Could not download {filename}: {e}")
                # Create dummy file with instructions
                dummy_content = f"""# Sample Data Placeholder
# Original source: {url}
# Please download manually or use data_loader.py to generate sample data
"""
                with open(filepath, 'w') as f:
                    f.write(dummy_content)
        
        print("\n✅ Sample data setup complete")
    
    def run_tests(self):
        """Run basic tests to verify setup"""
        self.print_header("RUNNING VERIFICATION TESTS")
        
        python = self.get_venv_python()
        
        tests = [
            ("Import pandas", "import pandas; print('✅ pandas')"),
            ("Import numpy", "import numpy; print('✅ numpy')"),
            ("Import sklearn", "import sklearn; print('✅ scikit-learn')"),
            ("Import tensorflow", "import tensorflow; print('✅ tensorflow')"),
            ("Import flask", "import flask; print('✅ flask')"),
            ("Import nltk", "import nltk; print('✅ nltk')"),
        ]
        
        all_passed = True
        for test_name, test_code in tests:
            try:
                result = subprocess.run(
                    [str(python), "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print(f"   ✅ {test_name}")
                else:
                    print(f"   ❌ {test_name}")
                    all_passed = False
            except Exception as e:
                print(f"   ❌ {test_name}: {e}")
                all_passed = False
        
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check installation.")
        
        return all_passed
    
    def print_next_steps(self):
        """Print instructions for next steps"""
        self.print_header("SETUP COMPLETE!")
        
        print("""
🎉 Your Sentiment Analysis environment is ready!

        """)
    
    
    
