"""
Weather data collector using OpenWeatherMap API
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import os
from ..utils.logger import log
from ..utils.database import DatabaseManager, WeatherData


class WeatherCollector:
    """
    Collects weather data from OpenWeatherMap API
    Supports both current weather and forecast data
    """

    def __init__(self, api_key: Optional[str] = None, config_path='configs/config.yaml'):
        """
        Initialize weather collector

        Args:
            api_key: OpenWeatherMap API key
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.db_manager = DatabaseManager(config_path)

    def _load_config(self, config_path):
        """Load configuration file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _load_api_key(self):
        """Load API key from api_keys.yaml"""
        api_keys_path = 'configs/api_keys.yaml'
        if os.path.exists(api_keys_path):
            with open(api_keys_path, 'r', encoding='utf-8') as f:
                keys = yaml.safe_load(f)
                return keys.get('openweathermap', {}).get('api_key')
        return None

    def get_current_weather(self, latitude: float, longitude: float, location_name: str) -> Optional[Dict]:
        """
        Fetch current weather data

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location

        Returns:
            Dictionary with weather data or None if failed
        """
        if not self.api_key:
            log.error("OpenWeatherMap API key not found!")
            return None

        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            weather_data = self._parse_current_weather(data, location_name, latitude, longitude)
            log.info(f"Successfully fetched current weather for {location_name}")
            return weather_data

        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching current weather: {e}")
            return None

    def get_forecast(self, latitude: float, longitude: float, location_name: str) -> Optional[List[Dict]]:
        """
        Fetch 5-day weather forecast (3-hour intervals)

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location

        Returns:
            List of weather data dictionaries or None if failed
        """
        if not self.api_key:
            log.error("OpenWeatherMap API key not found!")
            return None

        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            forecast_data = [
                self._parse_forecast_item(item, location_name, latitude, longitude)
                for item in data.get('list', [])
            ]
            log.info(f"Successfully fetched forecast for {location_name} ({len(forecast_data)} records)")
            return forecast_data

        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching forecast: {e}")
            return None

    def _parse_current_weather(self, data: Dict, location_name: str, lat: float, lon: float) -> Dict:
        """Parse current weather response"""
        return {
            'timestamp': datetime.fromtimestamp(data['dt']),
            'location_name': location_name,
            'latitude': lat,
            'longitude': lon,
            'temperature': data['main'].get('temp'),
            'humidity': data['main'].get('humidity'),
            'pressure': data['main'].get('pressure'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'wind_direction': data.get('wind', {}).get('deg'),
            'cloud_cover': data.get('clouds', {}).get('all'),
            'precipitation': data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
            'source': 'openweathermap'
        }

    def _parse_forecast_item(self, item: Dict, location_name: str, lat: float, lon: float) -> Dict:
        """Parse forecast item"""
        return {
            'timestamp': datetime.fromtimestamp(item['dt']),
            'location_name': location_name,
            'latitude': lat,
            'longitude': lon,
            'temperature': item['main'].get('temp'),
            'humidity': item['main'].get('humidity'),
            'pressure': item['main'].get('pressure'),
            'wind_speed': item.get('wind', {}).get('speed'),
            'wind_direction': item.get('wind', {}).get('deg'),
            'cloud_cover': item.get('clouds', {}).get('all'),
            'precipitation': item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0),
            'source': 'openweathermap'
        }

    def save_to_database(self, weather_data: Dict) -> bool:
        """
        Save weather data to database

        Args:
            weather_data: Dictionary with weather data

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.db_manager.get_session()
            weather_record = WeatherData(**weather_data)
            session.add(weather_record)
            session.commit()
            session.close()
            log.debug(f"Saved weather data for {weather_data['location_name']} at {weather_data['timestamp']}")
            return True
        except Exception as e:
            log.error(f"Error saving weather data to database: {e}")
            return False

    def collect_and_save(self, latitude: float, longitude: float, location_name: str, include_forecast: bool = True):
        """
        Collect weather data and save to database

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location
            include_forecast: Whether to include forecast data
        """
        # Collect current weather
        current = self.get_current_weather(latitude, longitude, location_name)
        if current:
            self.save_to_database(current)

        # Collect forecast
        if include_forecast:
            forecast = self.get_forecast(latitude, longitude, location_name)
            if forecast:
                for item in forecast:
                    self.save_to_database(item)

        log.info(f"Weather data collection completed for {location_name}")

    def collect_all_locations(self):
        """Collect weather data for all configured locations"""
        locations = self.config.get('data', {}).get('collection', {}).get('locations', [])

        if not locations:
            log.warning("No locations configured for data collection")
            return

        for location in locations:
            name = location.get('name')
            lat = location.get('latitude')
            lon = location.get('longitude')

            if name and lat and lon:
                log.info(f"Collecting weather data for {name}")
                self.collect_and_save(lat, lon, name)
            else:
                log.warning(f"Invalid location configuration: {location}")
