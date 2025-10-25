"""
Manual data collection script
Collects data from all configured sources
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.weather_collector import WeatherCollector
from src.data_collection.nasa_power_collector import NASAPowerCollector
from src.utils.logger import log


def collect_weather_data():
    """Collect weather data from OpenWeatherMap"""
    try:
        log.info("Collecting weather data...")
        collector = WeatherCollector()
        collector.collect_all_locations()
        log.info("Weather data collection completed!")
        return True
    except Exception as e:
        log.error(f"Error collecting weather data: {e}")
        return False


def collect_nasa_data(days_back=30):
    """
    Collect NASA POWER data

    Args:
        days_back: Number of days to collect from the past
    """
    try:
        log.info(f"Collecting NASA POWER data (last {days_back} days)...")
        collector = NASAPowerCollector()
        collector.collect_all_locations(days_back=days_back)
        log.info("NASA POWER data collection completed!")
        return True
    except Exception as e:
        log.error(f"Error collecting NASA POWER data: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect renewable energy forecast data')
    parser.add_argument(
        '--source',
        choices=['weather', 'nasa', 'all'],
        default='all',
        help='Data source to collect from'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days to collect NASA POWER data (default: 30)'
    )

    args = parser.parse_args()

    success = True

    if args.source in ['weather', 'all']:
        success = success and collect_weather_data()

    if args.source in ['nasa', 'all']:
        success = success and collect_nasa_data(args.days_back)

    if success:
        log.info("All data collection tasks completed successfully!")
    else:
        log.error("Some data collection tasks failed")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
