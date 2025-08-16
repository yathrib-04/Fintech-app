"""
Test script for Task 1 setup
Tests basic functionality and imports
"""

import sys
from pathlib import Path
import unittest

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestTask1Setup(unittest.TestCase):
    """Test basic setup and imports for Task 1"""
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            from scraping.playstore_scraper import BankingAppScraper
            from scraping.data_preprocessor import ReviewDataPreprocessor
            from scraping.config import BANK_APPS, QUALITY_THRESHOLDS
            print("✅ All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_config_structure(self):
        """Test configuration structure"""
        from scraping.config import BANK_APPS, QUALITY_THRESHOLDS
        
        # Check bank apps configuration
        self.assertIn('CBE', BANK_APPS)
        self.assertIn('BOA', BANK_APPS)
        self.assertIn('Dashen', BANK_APPS)
        
        # Check quality thresholds
        self.assertIn('target_total_reviews', QUALITY_THRESHOLDS)
        self.assertEqual(QUALITY_THRESHOLDS['target_total_reviews'], 1200)
        
        print("✅ Configuration structure correct")
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'data/raw',
            'data/processed',
            'src/scraping',
            'src/analysis',
            'src/database',
            'src/visualization',
            'tests'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Directory {dir_path} not found")
        
        print("✅ Directory structure correct")
    
    def test_file_existence(self):
        """Test that required files exist"""
        required_files = [
            'src/scraping/playstore_scraper.py',
            'src/scraping/data_preprocessor.py',
            'src/scraping/config.py',
            'src/scraping/run_task1.py',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        for file_path in required_files:
            self.assertTrue(Path(file_path).exists(), f"File {file_path} not found")
        
        print("✅ Required files exist")

def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running Task 1 Setup Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTask1Setup)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! Task 1 setup is ready.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
