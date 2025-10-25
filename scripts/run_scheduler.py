"""
Run the data collection scheduler
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.scheduler import run_scheduler

if __name__ == '__main__':
    run_scheduler()
