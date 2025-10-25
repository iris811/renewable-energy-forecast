"""
Database initialization script
Creates all necessary tables for the renewable energy forecast system
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.database import DatabaseManager
from src.utils.logger import log


def setup_database():
    """Initialize database and create all tables"""
    try:
        log.info("Starting database setup...")

        # Create database manager
        db_manager = DatabaseManager()

        # Create all tables
        db_manager.create_tables()

        log.info("Database setup completed successfully!")
        log.info(f"Database location: {db_manager.config.get('database', {}).get('sqlite', {}).get('path', './data/renewable_energy.db')}")

        return True

    except Exception as e:
        log.error(f"Error setting up database: {e}")
        return False


if __name__ == '__main__':
    success = setup_database()
    sys.exit(0 if success else 1)
