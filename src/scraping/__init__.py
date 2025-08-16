"""
Scraping module for Fintech App Analysis Project
Task 1: Data Collection and Preprocessing

This module contains all the scraping and data preprocessing functionality.
"""

from .playstore_scraper import BankingAppScraper
from .data_preprocessor import ReviewDataPreprocessor
from .config import BANK_APPS, QUALITY_THRESHOLDS, SCRAPING_SETTINGS

__version__ = "1.0.0"
__author__ = "Fintech App Analysis Team"

__all__ = [
    'BankingAppScraper',
    'ReviewDataPreprocessor',
    'BANK_APPS',
    'QUALITY_THRESHOLDS',
    'SCRAPING_SETTINGS'
]
