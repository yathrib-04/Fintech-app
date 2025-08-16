"""
Database module for Fintech App Analysis Project
Task 3: Store Cleaned Data in Oracle

This module contains all the database functionality including:
- Configuration management
- Schema definitions
- Database operations
- Data insertion and management
"""

from .config import DatabaseConfig, get_database_config
from .schema import DatabaseSchema
from .manager import DatabaseManager

__version__ = "1.0.0"
__author__ = "Fintech App Analysis Team"

__all__ = [
    'DatabaseConfig',
    'get_database_config',
    'DatabaseSchema',
    'DatabaseManager'
]
