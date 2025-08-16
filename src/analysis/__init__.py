"""
Analysis module for Fintech App Analysis Project
Task 2: Sentiment and Thematic Analysis

This module contains all the sentiment analysis and thematic analysis functionality.
"""

from .sentiment_analyzer import SentimentAnalyzer
from .theme_analyzer import ThemeAnalyzer

__version__ = "1.0.0"
__author__ = "Fintech App Analysis Team"

__all__ = [
    'SentimentAnalyzer',
    'ThemeAnalyzer'
]
