"""
Visualization module for Fintech App Analysis Project
Task 4: Insights and Recommendations

This module contains all the visualization and reporting functionality including:
- Chart generation and visualization
- Business insights analysis
- Report generation and formatting
- Complete Task 4 workflow orchestration
"""

from .chart_generator import ChartGenerator
from .insights_generator import InsightsGenerator
from .report_generator import ReportGenerator

__version__ = "1.0.0"
__author__ = "Fintech App Analysis Team"

__all__ = [
    'ChartGenerator',
    'InsightsGenerator',
    'ReportGenerator'
]
