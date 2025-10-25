"""
NASA POWER API data collector for solar radiation data
https://power.larc.nasa.gov/docs/services/api/
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import os
from ..utils.logger import log
from ..utils.database import DatabaseManager, WeatherData


class NASAPowerCollector:
    """
    Collects solar radiation and meteorological data from NASA POWER API
    Free to use, no API key required
    """

    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize NASA POWER collector

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.db_manager = DatabaseManager(config_path)

    def _load_config(self, config_path):
        """Load configuration file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def get_daily_data(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[Dict]]:
        """
        Fetch daily solar radiation and meteorological data

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            List of weather data dictionaries or None if failed
        """
        try:
            # NASA POWER parameters for renewable energy
            # ALLSKY_SFC_SW_DWN: All Sky Surface Shortwave Downward Irradiance (GHI)
            # ALLSKY_SFC_SW_DNI: Direct Normal Irradiance
            # ALLSKY_SFC_SW_DIFF: Diffuse Horizontal Irradiance
            # T2M: Temperature at 2 Meters
            # WS10M: Wind Speed at 10 Meters
            # RH2M: Relative Humidity at 2 Meters
            # PRECTOTCORR: Precipitation Corrected
            parameters = [
                'ALLSKY_SFC_SW_DWN',  # GHI
                'ALLSKY_SFC_SW_DNI',  # DNI
                'ALLSKY_SFC_SW_DIFF',  # DHI
                'T2M',                # Temperature
                'WS10M',              # Wind Speed
                'WS10M_MAX',          # Max Wind Speed
                'RH2M',               # Humidity
                'PRECTOTCORR',        # Precipitation
                'PS',                 # Surface Pressure
                'CLOUD_AMT'           # Cloud Amount
            ]

            params = {
                'parameters': ','.join(parameters),
                'community': 'RE',  # Renewable Energy community
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.strftime('%Y%m%d'),
                'end': end_date.strftime('%Y%m%d'),
                'format': 'JSON'
            }

            log.info(f"Fetching NASA POWER data for {location_name} from {start_date.date()} to {end_date.date()}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'properties' not in data or 'parameter' not in data['properties']:
                log.error("Invalid response from NASA POWER API")
                return None

            weather_data_list = self._parse_nasa_data(
                data['properties']['parameter'],
                location_name,
                latitude,
                longitude
            )

            log.info(f"Successfully fetched {len(weather_data_list)} records from NASA POWER for {location_name}")
            return weather_data_list

        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching NASA POWER data: {e}")
            return None
        except Exception as e:
            log.error(f"Error parsing NASA POWER data: {e}")
            return None

    def _parse_nasa_data(
        self,
        parameters: Dict,
        location_name: str,
        latitude: float,
        longitude: float
    ) -> List[Dict]:
        """
        Parse NASA POWER API response

        Args:
            parameters: Parameter data from API response
            location_name: Name of the location
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            List of weather data dictionaries
        """
        weather_data_list = []

        # Get all dates from the first parameter
        first_param = next(iter(parameters.values()))
        dates = list(first_param.keys())

        for date_str in dates:
            try:
                # Parse date (format: YYYYMMDD)
                date = datetime.strptime(date_str, '%Y%m%d')

                # Extract values (handle missing data marked as -999)
                def get_value(param_name):
                    value = parameters.get(param_name, {}).get(date_str)
                    return None if value == -999 or value is None else value

                weather_data = {
                    'timestamp': date,
                    'location_name': location_name,
                    'latitude': latitude,
                    'longitude': longitude,
                    'temperature': get_value('T2M'),
                    'humidity': get_value('RH2M'),
                    'pressure': get_value('PS'),  # kPa, need to convert to hPa
                    'wind_speed': get_value('WS10M'),
                    'wind_direction': None,  # Not provided by NASA POWER
                    'cloud_cover': get_value('CLOUD_AMT'),
                    'precipitation': get_value('PRECTOTCORR'),
                    'solar_irradiance': get_value('ALLSKY_SFC_SW_DWN'),  # GHI
                    'ghi': get_value('ALLSKY_SFC_SW_DWN'),
                    'dni': get_value('ALLSKY_SFC_SW_DNI'),
                    'dhi': get_value('ALLSKY_SFC_SW_DIFF'),
                    'source': 'nasa_power'
                }

                # Convert pressure from kPa to hPa if available
                if weather_data['pressure'] is not None:
                    weather_data['pressure'] = weather_data['pressure'] * 10

                weather_data_list.append(weather_data)

            except Exception as e:
                log.warning(f"Error parsing date {date_str}: {e}")
                continue

        return weather_data_list

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
            log.debug(f"Saved NASA POWER data for {weather_data['location_name']} at {weather_data['timestamp']}")
            return True
        except Exception as e:
            log.error(f"Error saving NASA POWER data to database: {e}")
            return False

    def collect_and_save(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        days_back: int = 30
    ):
        """
        Collect NASA POWER data and save to database

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Name of the location
            days_back: Number of days to collect from the past
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        data_list = self.get_daily_data(latitude, longitude, location_name, start_date, end_date)

        if data_list:
            for data in data_list:
                self.save_to_database(data)
            log.info(f"Saved {len(data_list)} NASA POWER records for {location_name}")

    def collect_all_locations(self, days_back: int = 30):
        """
        Collect NASA POWER data for all configured locations

        Args:
            days_back: Number of days to collect from the past
        """
        locations = self.config.get('data', {}).get('collection', {}).get('locations', [])

        if not locations:
            log.warning("No locations configured for data collection")
            return

        for location in locations:
            name = location.get('name')
            lat = location.get('latitude')
            lon = location.get('longitude')

            if name and lat and lon:
                log.info(f"Collecting NASA POWER data for {name}")
                self.collect_and_save(lat, lon, name, days_back)
            else:
                log.warning(f"Invalid location configuration: {location}")
